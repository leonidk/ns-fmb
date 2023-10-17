"""
Nerfstudio Fuzzy Metaball Config

A custom method that implements Fuzzy Metaballs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from method_fmb.fmb_model import FMBModelConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig, OptimizerConfig 
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

method_fmb = MethodSpecification(
    config=TrainerConfig(
        method_name="fmb", 
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=1024,
            ),
            model=FMBModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-6),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=50000),
            },
            "precs": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-6),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=50000),
            },
            "wlog": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-6),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=50000),
            },
            "colors": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-6),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=50000),
            },
            "background": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=50000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio Fuzzy Metaball method.",
)
