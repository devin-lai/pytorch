# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# Tests that backward hooks attached to stage-boundary activations do not
# accumulate across pipeline steps. Regression test for:
# https://github.com/pytorch/pytorch/issues/185331

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.pipelining import Schedule1F1B, SplitPoint, pipeline
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_MULTIACCELERATOR,
)

try:
    import torch.accelerator
    acc = torch.accelerator.current_accelerator()
    device_type = acc.type if acc else "cpu"
except Exception:
    device_type = "cpu"

backend = dist.get_default_backend_for_device(device_type)


class _TwoLayerModel(nn.Module):
    """Simple two-layer model for pipeline split testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _add_tensor_hook(module, fired_hook_ids, next_hook_id):
    """
    Attach a module forward pre-hook that registers a tensor backward hook
    on the first input. This pattern is common in debugging and gradient
    collection tools. The bug was that these tensor hooks accumulated on the
    persistent recv buffer across steps.
    """
    def pre_hook(_module, inputs):
        x = inputs[0]
        hook_id = next_hook_id[0]
        next_hook_id[0] += 1

        def tensor_hook(_grad):
            fired_hook_ids.append(hook_id)

        x.register_hook(tensor_hook)

    module.register_forward_pre_hook(pre_hook)


class RecvBufferHookTest(MultiProcContinuousTest):
    """
    Verify that backward hooks attached to pipeline stage-boundary
    activations do not persist across steps.

    Before the fix, recv buffers were directly aliased as stage input
    activations via requires_grad_() which is in-place and returns self.
    Hooks attached in step N persisted into step N+1 because the same
    tensor instance was reused, causing fired hook counts to grow as
    2, 4, 6, 8, 10 instead of staying constant.

    The fix uses detach() to create a new tensor identity each step
    while sharing the buffer's storage (zero-copy).  This preserves
    pinned memory or custom allocator properties of the recv buffer
    while giving the activation an independent lifecycle so hooks are
    naturally discarded when the activation goes out of scope.
    """

    world_size = 2

    @classmethod
    def backend_str(cls) -> str:
        return backend

    @classmethod
    def device_type(cls) -> str:
        return device_type

    @property
    def device(self) -> torch.device:
        if device_type == "cpu":
            return torch.device("cpu")
        return torch.device(device_type, self.rank)

    @requires_accelerator_dist_backend(["nccl", "xccl", "gloo"])
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR and device_type != "cpu",
        "Test requires accelerator or CPU",
    )
    def test_hooks_do_not_accumulate_across_steps(self):
        """
        Hook fired count must stay constant (== n_microbatches) across all
        training steps. Before the fix it grew by n_microbatches each step.

        Uses detach() instead of clone() so the recv buffer's storage is
        shared (zero-copy) and any pinned/custom allocator properties are
        preserved, while still giving each step a fresh tensor identity so
        hooks do not accumulate.
        """
        n_microbatches = 2
        n_steps = 5
        device = self.device

        model = _TwoLayerModel().to(device)
        sample = torch.zeros(n_microbatches, 4, device=device)

        pipe = pipeline(
            module=model,
            mb_args=(sample,),
            split_spec={"fc1": SplitPoint.END},
        )
        stage = pipe.build_stage(self.rank, device, None)
        schedule = Schedule1F1B(
            stage, n_microbatches=n_microbatches, loss_fn=F.mse_loss
        )

        fired_hook_ids: list[int] = []
        next_hook_id = [0]

        if self.rank == 1:
            # Hook fc2 on the receiving stage. Before the fix, fc2's input
            # tensor was the persistent recv buffer, so hooks accumulated.
            for name, module in stage.submod.named_modules():
                if name == "fc2" or name.endswith(".fc2"):
                    _add_tensor_hook(module, fired_hook_ids, next_hook_id)
                    break

        hooks_per_step: list[int] = []

        for _step in range(n_steps):
            fired_hook_ids.clear()
            x = torch.randn(n_microbatches, 4, device=device)
            target = torch.randn(n_microbatches, 4, device=device)

            if self.rank == 0:
                schedule.step(x, return_outputs=False)
            else:
                schedule.step(target=target, return_outputs=False)

            if self.rank == 1:
                hooks_per_step.append(len(fired_hook_ids))

        if self.rank == 1:
            # Every step must fire exactly n_microbatches hooks.
            # Before the fix (buffer aliased as activation): [2, 4, 6, 8, 10]
            # After the fix  (detach gives fresh identity):  [2, 2, 2, 2,  2]
            expected = n_microbatches
            for step, count in enumerate(hooks_per_step):
                self.assertEqual(
                    count,
                    expected,
                    msg=(
                        f"Step {step}: expected {expected} hooks fired "
                        f"(== n_microbatches), got {count}. "
                        f"Hook accumulation regression detected. "
                        f"All steps: {hooks_per_step}"
                    ),
                )


instantiate_parametrized_tests(RecvBufferHookTest)

if __name__ == "__main__":
    run_tests()
