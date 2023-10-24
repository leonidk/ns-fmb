"""
Nerfstudio Fuzzy Metaball Config

A custom method that implements Fuzzy Metaballs.
"""

from __future__ import annotations



from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
)
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipelineConfig,
)
from method_fmb.fmb_model import FMBModelConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig 
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

method_fmb = MethodSpecification(
    config=TrainerConfig(
        method_name="fmb", 
        steps_per_eval_batch=500,
        steps_per_save=1000,
        max_num_iterations=3000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(downscale_factor=8),
                train_num_rays_per_batch=1<<13,
                eval_num_rays_per_batch=1<<13,
            ),
            model=FMBModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=6e-3, eps=1e-15, weight_decay=5e-6),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-7, max_steps=55000),
            },
            "precs": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=2e-6),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=2e-7, max_steps=55000),
            },
            "wlog": {
                "optimizer": AdamOptimizerConfig(lr=3e-3, eps=1e-15, weight_decay=2e-6),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=4e-8, max_steps=55000),
            },
            "colors": {
                "optimizer": AdamOptimizerConfig(lr=2e-2, eps=1e-15, weight_decay=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=3e-7, max_steps=55000),
            },
            "background": {
                "optimizer": AdamOptimizerConfig(lr=3e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=4e-8, max_steps=55000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio Fuzzy Metaball method.",
)
