"""CPU contract tests; these do not establish learned or real-robot performance.

Run: python -m unittest openpi.baselines.touchworld_lite_test
"""

import inspect
import unittest

import torch

from openpi.baselines.touchworld_lite import TouchWorldLite, TouchWorldLiteConfig


class TouchWorldLiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9)
        self.cfg = TouchWorldLiteConfig(
            num_tasks=3, action_horizon=4, window_size=3, history_frames=3,
            width=32, latent_dim=16, image_grid=2, num_heads=4,
            goal_layers=1, nominal_layers=1, residual_layers=1,
        )
        self.model = TouchWorldLite(self.cfg)
        self.current = dict(
            current_rgb=torch.rand(2, 3, 16, 16) * 2 - 1,
            current_tactile=torch.randn(2, 3, 5, 120, 3),
            current_state=torch.randn(2, 18),
            task_id=torch.tensor([0, 1]),
        )
        self.targets = dict(
            actions=torch.randn(2, 4, 18),
            future_rgb=torch.rand(2, 3, 16, 16) * 2 - 1,
            future_tactile=torch.randn(2, 5, 120, 3),
            future_valid_mask=torch.tensor([True, True]),
            action_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        )

    def _prepare_residual(self):
        self.model.freeze_observation_encoders().freeze_nominal()
        self.plan = self.model.predict_nominal(**self.current)
        self.refinement = dict(
            plan=self.plan,
            current_tactile=torch.randn_like(self.current["current_tactile"]),
            current_state=torch.randn_like(self.current["current_state"]),
            offsets=torch.tensor([1, 3]),
        )

    def test_observation_stage_reconstructs_and_only_updates_autoencoders(self):
        losses = self.model(
            stage="representation", rgb=self.current["current_rgb"], tactile=self.current["current_tactile"]
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.observation_ae.parameters()))
        self.assertTrue(all(p.grad is None for n, p in self.model.named_parameters() if not n.startswith("observation_ae.")))
        with self.assertRaises(RuntimeError):
            self.model.predict_nominal(**self.current)

    def test_both_future_goals_are_predicted_with_required_task_condition(self):
        self.model.freeze_observation_encoders()
        plan = self.model.predict_nominal(**self.current)
        self.assertEqual(plan.actions.shape, (2, 4, 18))
        self.assertEqual(plan.predicted_visual_goal.shape, (2, 4, 16))
        self.assertEqual(plan.predicted_tactile_goal.shape, (2, 5, 16))
        self.assertFalse(plan.actions.requires_grad)
        changed = dict(self.current, task_id=torch.tensor([2, 2]))
        self.assertFalse(torch.allclose(plan.actions, self.model.predict_nominal(**changed).actions))
        missing_task = {k: v for k, v in self.current.items() if k != "task_id"}
        with self.assertRaises(TypeError):
            self.model.predict_nominal(**missing_task)
        self.assertNotIn("future_rgb", inspect.signature(self.model.predict_nominal).parameters)
        self.assertNotIn("future_tactile", inspect.signature(self.model.refine).parameters)

    def test_future_labels_supervise_but_never_condition_actions(self):
        self.model.freeze_observation_encoders().train()
        first = self.model.nominal_loss(**self.current, **self.targets)
        altered_targets = dict(
            self.targets, future_rgb=-self.targets["future_rgb"],
            future_tactile=self.targets["future_tactile"] + 2,
        )
        second = self.model.nominal_loss(**self.current, **altered_targets)
        torch.testing.assert_close(first["action_loss"], second["action_loss"], rtol=0, atol=0)
        self.assertFalse(torch.allclose(first["tactile_goal_loss"], second["tactile_goal_loss"]))
        self.assertFalse(torch.allclose(first["visual_goal_loss"], second["visual_goal_loss"]))
        before = {n: p.detach().clone() for n, p in self.model.observation_ae.named_parameters()}
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)
        first["loss"].backward()
        self.assertGreater(self.model.visual_goal_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.tactile_goal_head.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        for name, parameter in self.model.observation_ae.named_parameters():
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertFalse(self.model.observation_ae.training)

    def test_nominal_action_objective_reads_both_predicted_goals(self):
        self.model.freeze_observation_encoders()
        # Use only action supervision here: both goal heads must participate in
        # the policy, independently of their separate future-prediction losses.
        self.model.nominal_loss(**self.current, **self.targets)["action_loss"].backward()
        for head in (self.model.visual_goal_head, self.model.tactile_goal_head):
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)

    def test_nominal_masked_padding_nan_is_ignored_and_targets_detached(self):
        self.model.freeze_observation_encoders()
        targets = {k: v.clone() for k, v in self.targets.items()}
        targets["future_valid_mask"][1] = False
        targets["action_valid_mask"][1, 2:] = False
        baseline = self.model.nominal_loss(**self.current, **targets)
        targets["future_rgb"][1] = float("nan")
        targets["future_tactile"][1] = float("inf")
        targets["actions"][1, 2:] = float("nan")
        for key in ("future_rgb", "future_tactile", "actions"):
            targets[key].requires_grad_()
        masked = self.model.nominal_loss(**self.current, **targets)
        torch.testing.assert_close(masked["loss"], baseline["loss"], rtol=0, atol=0)
        masked["loss"].backward()
        for key in ("future_rgb", "future_tactile", "actions"):
            self.assertIsNone(targets[key].grad)

    def test_zero_initial_residual_is_identity_on_actual_predicted_plan(self):
        self._prepare_residual()
        output = self.model.refine(**self.refinement)
        torch.testing.assert_close(output[0], self.plan.actions[0, 1:4], rtol=0, atol=0)
        torch.testing.assert_close(output[1, 0], self.plan.actions[1, 3], rtol=0, atol=0)
        loss = self.model.residual_loss(
            **self.refinement, target_actions=self.targets["actions"],
            action_valid_mask=self.targets["action_valid_mask"],
        )
        error = torch.cat([
            self.plan.actions[0, 1:4, 6:] - self.targets["actions"][0, 1:4, 6:],
            self.plan.actions[1, 3:4, 6:] - self.targets["actions"][1, 3:4, 6:],
        ])
        torch.testing.assert_close(loss["residual_loss"], error.square().mean())
        self.assertEqual(loss["valid_actions"].item(), 4)

    def test_residual_uses_latest_touch_and_state_preserving_unselected_actions(self):
        self._prepare_residual()
        with torch.no_grad():
            self.model.residual_head.weight.normal_(std=0.02)
        original_plan = self.plan.actions.clone()
        output = self.model.refine(**self.refinement)
        changed_touch = dict(self.refinement, current_tactile=self.refinement["current_tactile"] + 3)
        changed_state = dict(self.refinement, current_state=self.refinement["current_state"] + 3)
        self.assertFalse(torch.allclose(output, self.model.refine(**changed_touch)))
        self.assertFalse(torch.allclose(output, self.model.refine(**changed_state)))
        torch.testing.assert_close(output[0, :, :6], self.plan.actions[0, 1:4, :6], rtol=0, atol=0)
        torch.testing.assert_close(self.plan.actions, original_plan, rtol=0, atol=0)
        # No accumulation: repeat calls against the same original plan are identical.
        torch.testing.assert_close(output, self.model.refine(**self.refinement), rtol=0, atol=0)

    def test_residual_gradients_stay_in_refiner_and_past_labels_are_ignored(self):
        self._prepare_residual()
        self.model.train()
        targets = self.targets["actions"].clone()
        valid = self.targets["action_valid_mask"].clone()
        valid[0, 3] = False
        baseline = self.model.residual_loss(**self.refinement, target_actions=targets, action_valid_mask=valid)
        targets[0, 0] = float("nan")  # already executed
        targets[0, 3] = float("inf")  # padded
        targets[1, :3] = float("nan")  # already executed
        masked = self.model.residual_loss(**self.refinement, target_actions=targets, action_valid_mask=valid)
        torch.testing.assert_close(masked["loss"], baseline["loss"], rtol=0, atol=0)
        masked["loss"].backward()
        self.assertGreater(self.model.residual_head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for n, p in self.model.named_parameters() if not n.startswith("residual_")))
        self.assertFalse(self.model.observation_ae.training)
        self.assertFalse(self.model.nominal_decoder.training)

    def test_all_masked_losses_are_finite_zero(self):
        self.model.freeze_observation_encoders()
        targets = dict(
            self.targets, future_valid_mask=torch.zeros(2, dtype=torch.bool),
            action_valid_mask=torch.zeros(2, 4, dtype=torch.bool),
            future_rgb=torch.full_like(self.targets["future_rgb"], float("nan")),
            future_tactile=torch.full_like(self.targets["future_tactile"], float("nan")),
            actions=torch.full_like(self.targets["actions"], float("nan")),
        )
        loss = self.model.nominal_loss(**self.current, **targets)["loss"]
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self._prepare_residual()
        residual_loss = self.model.residual_loss(
            **self.refinement, target_actions=targets["actions"], action_valid_mask=targets["action_valid_mask"]
        )["loss"]
        self.assertEqual(residual_loss.item(), 0)
        residual_loss.backward()
        self.assertTrue(torch.isfinite(self.model.residual_head.weight.grad).all())

    def test_stage_checkpoint_restores_freezing_and_validates_offset(self):
        self._prepare_residual()
        restored = TouchWorldLite(self.cfg)
        restored.load_state_dict(self.model.state_dict())
        restored.train()
        self.assertEqual(restored.training_stage, "residual")
        self.assertTrue(all(p.requires_grad == n.startswith("residual_") for n, p in restored.named_parameters()))
        with self.assertRaises(ValueError):
            restored.refine(**dict(self.refinement, offsets=torch.tensor([0, 4])))
        with self.assertRaises(ValueError):
            restored.refine(**dict(self.refinement, offsets=torch.tensor([0.0, 1.0])))
        with self.assertRaises(RuntimeError):
            restored(stage="nominal", **self.current, **self.targets)


if __name__ == "__main__":
    unittest.main()
