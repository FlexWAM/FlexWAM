"""Config checks against the tasks shipped in configs/task."""
import importlib
from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from flexwam.config import compose_task, resolve_config
from flexwam.models.architectures.view_scoring_head import arches_compatible, normalize_head_arch
from flexwam.models.architectures.critic import CameraCritic
from flexwam.models.architectures.router import CameraRouter
from flexwam.training.trainer import FlexWAMTrainer
from flexwam.utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[1]
LINGBOT_TASK = 'lingbot/flexiv_cable'


class ConfigTests(unittest.TestCase):
    def public(self, task=LINGBOT_TASK, overrides=()):
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
            return compose(config_name='train', overrides=[f'task={task}', *overrides])

    def test_new_dataset_reuses_robot_contract(self):
        cfg = resolve_config(self.public(overrides=['data.roots=[/new/dataset]', 'train.batch_size=3']))
        self.assertEqual(list(cfg.data.train.dataset_dirs), ['/new/dataset'])
        self.assertEqual(list(cfg.data.val.dataset_dirs), ['/new/dataset'])
        self.assertEqual(cfg.batch_size, 3)
        self.assertEqual(cfg.data.train.processor.num_output_cameras, 3)
        self.assertEqual(list(cfg.data.train.video_size), [192,224])
        self.assertEqual(list(cfg.data.train.processor.train_transforms[0].size), [192,224])

    def test_video_state_is_validated_and_heads_reject_state(self):
        for task in ('wan/piper_pencil_case', 'lingbot/flexiv_cable'):
            with self.subTest(task=task):
                cfg = resolve_config(self.public(task))
                expected_shift = 5.0 if task.startswith('lingbot/') else 1.0
                self.assertEqual(cfg.model.action_scheduler.train_shift, expected_shift)
                self.assertEqual(cfg.model.action_scheduler.infer_shift, expected_shift)
                self.assertEqual(cfg.model.video_scheduler.train_shift, 5.0)
                self.assertNotIn('state_conditioning', cfg.model.action_router)
                self.assertNotIn('state_conditioning', cfg.model.action_critic)
        cfg = resolve_config(self.public('wan/piper_pencil_case', ['model.video.state_conditioning.enabled=true']))
        self.assertTrue(cfg.model.video_dit_config.state_conditioning.enabled)
        self.assertNotIn('drop_prob', cfg.model.state_conditioning)
        with self.assertRaises(ValueError):
            resolve_config(self.public('wan/piper_pencil_case', ['model.video.state_conditioning.num_tokens=2']))
        public = self.public('wan/piper_pencil_case')
        for label in ('router', 'critic'):
            with self.subTest(label=label):
                broken = OmegaConf.create(OmegaConf.to_container(public, resolve=False))
                with open_dict(broken.model[label]):
                    broken.model[label].state_conditioning = {'enabled': True, 'mode': 'late_concat', 'hidden_dim': 512}
                with self.assertRaisesRegex(ValueError, 'state_conditioning'):
                    resolve_config(broken)

    def test_two_camera_task_and_critic_variant(self):
        public = self.public()
        del public.data.observation.cameras[1]
        cfg = resolve_config(public)
        self.assertEqual(cfg.model.video_dit_config.num_views, 2)
        self.assertEqual(cfg.model.action_router.num_views, 2)
        self.assertEqual(cfg.model.action_critic.num_views, 2)
        self.assertNotIn('same_router', cfg.model.action_critic)
        self.assertEqual(cfg.model.action_critic.feature_layer_index, cfg.model.action_router.feature_layer_index)
        self.assertEqual(compose_task(LINGBOT_TASK).model.router_cf.intervention, 'router_zero')
        overridden = compose_task(LINGBOT_TASK, overrides=('train.counterfactual.intervention=pixel',))
        self.assertEqual(overridden.model.router_cf.intervention, 'pixel')
        self.assertNotIn('probe', overridden.model.router_cf)

    def test_critic_reads_its_own_config(self):
        cfg = compose_task('lingbot/flexiv_cable')
        self.assertEqual(cfg.model.action_critic.arch, 'spatial_query_v1')
        self.assertEqual(cfg.model.action_critic.feature_dim, cfg.model.action_router.feature_dim)
        overridden = compose_task('lingbot/flexiv_cable', overrides=(
            'model.critic.feature_layer_index=20',
            'model.critic.feature_dim=256',
        ))
        self.assertEqual(overridden.model.action_critic.feature_layer_index, 20)
        self.assertEqual(overridden.model.action_critic.feature_dim, 256)
        self.assertEqual(overridden.model.action_router.feature_layer_index, 18)
        self.assertEqual(overridden.model.action_router.feature_dim, 512)

    def test_missing_critic_copies_router_network(self):
        public = self.public('lingbot/flexiv_cable')
        del public.model.critic
        cfg = resolve_config(public)
        self.assertNotIn('same_router', cfg.model.action_critic)
        self.assertEqual(cfg.model.action_critic.arch, 'spatial_query_v1')
        self.assertEqual(cfg.model.action_critic.feature_layer_index, cfg.model.action_router.feature_layer_index)
        self.assertEqual(cfg.model.action_critic.feature_dim, cfg.model.action_router.feature_dim)

    def test_legacy_head_arch_alias_and_independent_modules(self):
        cfg = compose_task('lingbot/flexiv_cable', overrides=('model.router.arch=l18_spatial_query_v1',))
        self.assertEqual(cfg.model.action_router.arch, 'spatial_query_v1')
        self.assertEqual(cfg.model.action_critic.arch, 'spatial_query_v1')
        self.assertTrue(arches_compatible('l18_spatial_query_v1', 'spatial_query_v1'))
        self.assertEqual(normalize_head_arch('l18'), 'spatial_query_v1')
        kwargs = dict(
            token_dim=8, num_views=2, feature_dim=8, pool_grid=[2, 2],
            num_queries=2, num_heads=2, spatial_hidden_dim=16,
            branch_output_dim=8, head_hidden_dim=16,
        )
        router, critic = CameraRouter(**kwargs), CameraCritic(**kwargs)
        self.assertIsNot(router, critic)
        self.assertTrue(all(id(left) != id(right) for left, right in zip(router.parameters(), critic.parameters())))

    def test_router_overrides_do_not_change_critic(self):
        cfg = compose_task('lingbot/flexiv_cable', overrides=(
            'model.router.feature_layer_index=20',
            'model.router.feature_dim=256',
        ))
        self.assertEqual(cfg.model.action_router.feature_layer_index, 20)
        self.assertEqual(cfg.model.action_router.feature_dim, 256)
        self.assertEqual(cfg.model.action_critic.feature_layer_index, 18)
        self.assertEqual(cfg.model.action_critic.feature_dim, 512)

    def test_bad_camera_or_layer_config_fails_early(self):
        for overrides in [
            ['model.video.num_views=2'],
            ['model.router.feature_layer_index=24'],
            ['model.critic.feature_layer_index=24'],
            ['data.sampling.video_stride=3'],
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_config(self.public(overrides=overrides))

    def test_model_factory_and_processor_targets_resolve(self):
        cfg = compose_task('lingbot/flexiv_cable')
        for target in [cfg.model._target_,cfg.data.train.processor._target_]:
            module,name=target.rsplit('.',1)
            self.assertTrue(callable(getattr(importlib.import_module(module), name)))

    def test_validation_uses_training_normalization(self):
        for mode in ['q01/q99', 'z-score']:
            with self.subTest(mode=mode):
                cfg = resolve_config(self.public('wan/piper_pencil_case', [f'data.normalization.mode={mode}']))
                self.assertEqual(cfg.data.train.processor.norm_default_mode, mode)
                self.assertEqual(cfg.data.val.processor.norm_default_mode, mode)

    def test_runtime_projection_is_idempotent(self):
        cfg=compose_task('lingbot/flexiv_cable')
        self.assertIs(resolve_config(cfg),cfg)

    def test_no_router_ablation_keeps_heads_off_under_default_stages(self):
        for task in ('ablation/no_router',):
            with self.subTest(task=task):
                cfg = compose_task(task)
                self.assertFalse(cfg.model.action_router.enabled)
                self.assertFalse(cfg.model.action_critic.enabled)
                stages = cfg.training_stages
                self.assertTrue(stages.enabled)
                self.assertEqual(float(stages.stage1_ratio), 1.0)
                self.assertEqual(float(stages.stage2_ratio), 0.0)
                self.assertEqual(float(stages.stage3_ratio), 0.0)
                self.assertEqual(float(stages.stage4_ratio), 0.0)
                bounds = FlexWAMTrainer._stage_step_bounds(
                    FlexWAMTrainer._stage_fractions(stages), 13470
                )
                self.assertEqual(bounds, [0, 13470, 13470, 13470, 13470, 13470])

    def test_router_no_cf_ablation_trains_router_without_critic_stages(self):
        for task in ('ablation/no_cf',):
            with self.subTest(task=task):
                cfg = compose_task(task)
                self.assertTrue(cfg.model.action_router.enabled)
                self.assertFalse(cfg.model.action_critic.enabled)
                stages = cfg.training_stages
                self.assertTrue(stages.enabled)
                self.assertEqual(float(stages.stage1_ratio), 0.1)
                self.assertEqual(float(stages.stage2_ratio), 0.9)
                self.assertEqual(float(stages.stage3_ratio), 0.0)
                self.assertEqual(float(stages.stage4_ratio), 0.0)
                bounds = FlexWAMTrainer._stage_step_bounds(
                    FlexWAMTrainer._stage_fractions(stages), 10000
                )
                self.assertEqual(bounds, [0, 1000, 10000, 10000, 10000, 10000])

    def test_direct_cf_ablation_skips_critic_and_uses_router_zero(self):
        cfg = compose_task('ablation/cf_direct')
        self.assertTrue(cfg.model.action_router.enabled)
        self.assertFalse(cfg.model.action_critic.enabled)
        self.assertEqual(cfg.model.router_cf.teacher, 'cf')
        self.assertEqual(cfg.model.router_cf.intervention, 'router_zero')
        self.assertEqual(cfg.model.action_attention.mode, 'logit_bias')
        self.assertEqual(int(cfg.model.video_dit_config.cross_view_attention_interval), 2)
        stages = cfg.training_stages
        self.assertEqual(float(stages.stage1_ratio), 0.1)
        self.assertEqual(float(stages.stage2_ratio), 0.5)
        self.assertEqual(float(stages.stage3_ratio), 0.0)
        self.assertEqual(float(stages.stage4_ratio), 0.4)
        self.assertEqual(float(stages.stage5_ratio), 0.0)
        bounds = FlexWAMTrainer._stage_step_bounds(
            FlexWAMTrainer._stage_fractions(stages), 10000
        )
        self.assertEqual(bounds, [0, 1000, 6000, 6000, 10000, 10000])


if __name__ == '__main__':
    unittest.main()
