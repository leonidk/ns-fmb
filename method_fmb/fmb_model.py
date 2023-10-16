"""
Fuzzy Metaballs Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type, Literal

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.utils import colormaps

@dataclass
class FMBModelConfig(ModelConfig):
    """Fuzzy Metaballs Model Configuration."""

    _target: Type = field(default_factory=lambda: FMBModel)

    # init settings
    num_gaussians: int = 512
    mean_dist: float = 1e-3
    cov_dist: float = 80
    w_scale: float = 0.5

    # loss settings
    beta_loss_scale: float = 0.0

    # render settings
    use_two_param: bool = True
    beta1: float = 24.4
    beta2: float = 3.14


class FMBModel(Model):
    """Fuzzy Metaballs Model."""

    config: FMBModelConfig

    def __init__(
        self,
        config: FMBModelConfig,
        **kwargs,
    ) -> None:
        self.means = None
        self.precs = None
        self.wlogs = None
        self.colors = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        super().populate_modules()
        
        # init
        num_g = self.config.num_gaussians
        tmp_dist = MultivariateNormal(torch.zeros(3),torch.eye(3))
        self.means = Parameter(self.config.mean_dist*tmp_dist.sample((num_g,)))
        self.precs = Parameter(self.config.cov_dist*torch.tile(torch.eye(3),(num_g,1,)).reshape((-1,3,3)))
        self.wlogs = Parameter(torch.log(self.config.w_scale*torch.ones(num_g)))
        self.colors = Parameter(torch.randn((num_g,3)))
        self.bg_color = Parameter(torch.randn(3))

        # losses
        self.rgb_loss = MSELoss()

        # renderers
        #self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        #self.renderer_accumulation = AccumulationRenderer()
        #self.renderer_depth = DepthRenderer(method="median")
        #self.renderer_expected_depth = DepthRenderer(method="expected")
        #self.renderer_normals = NormalsRenderer()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.means is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        param_groups["means"] =[self.means] 
        param_groups["precs"] =[self.precs]
        param_groups["wlog"] =[self.wlogs]
        param_groups["colors"] =[self.colors,self.bg_color]
        return param_groups
 

    def get_outputs(self, ray_bundle: RayBundle):
        if self.means is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        
        tprec = torch.transpose(torch.triu(self.precs),1,2)

        # basic quantities compute
        tt = ray_bundle.origins
        tr = ray_bundle.directions

        p = self.means[None] - tt[:,None]

        projp = torch.einsum('kij,bkj->bki',tprec,p)
        projr = torch.einsum('kij,bj->bki',tprec,tr)
        vsv = (projr**2).sum(axis=2)
        psv = (projp*projr).sum(axis=2)
        z = psv/vsv
        v = tr[:,None] * z[:,:,None] - p

        # distance
        d0 = (torch.einsum('kij,bkj->bki',tprec,v)**2).sum(axis=2)
        d2 = -0.5*d0 + self.wlogs[None]

        # normal 
        projp2 = torch.einsum('kji,bkj->bki',tprec,projp)
        projp2_norm = torch.linalg.norm(projp2,dim=2,keepdims=True)
        norm_b_est = projp2/projp2_norm
        norm_sign = -torch.sign(torch.einsum('bki,bi->bk',norm_b_est,tr))
        norm_est = norm_sign[:,:,None] * norm_b_est

        # combine
        sig1 = (z > 0)
        d2_z = torch.nan_to_num(d2/sig1)

        if self.config.use_two_param:
            est_alpha = 1-torch.exp(-torch.exp(d2_z).sum(axis=1))
            est_logit = -z*self.config.beta1 + self.config.beta2*d2
            w = sig1*torch.softmax(est_logit,1)+1e-46
        else:
            densities = torch.exp(d2_z)
            zidx = torch.argsort(z,axis=1)#descending=True)
            order_density = torch.gather(densities,1,zidx)
            order_summed_density = torch.cumsum(order_density,1)
            order_prior_density =  order_summed_density - order_density
            est_alpha = 1 - torch.exp(-order_summed_density[:,-1])
            prior_density = torch.zeros_like(order_prior_density)
            prior_density = torch.scatter(order_prior_density,1,zidx,order_prior_density)
            transmit = torch.exp(-prior_density)
            w = transmit * (1-torch.exp(-densities)) + 1e-20
        wsum = w.sum(1,keepdims=True)
        w = w/torch.where(wsum!=0,wsum,1)
        final_z = (w*z).sum(axis=1)
        final_norm = (w[:,:,None]*norm_est).sum(axis=1)
        final_norm = final_norm/torch.linalg.norm(final_norm,axis=1,keepdims=True)

        pad_alpha = est_alpha[:,None]
        obj_color = w @ torch.sigmoid(self.colors)
        final_color = (1-pad_alpha) * torch.sigmoid(self.bg_color) + pad_alpha * obj_color

        outputs = {
            "rgb": final_color,
            "accumulation": est_alpha,
            "depth": final_z,
            "normals": final_norm,
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)
        rgb_loss_fine = torch.abs(image - outputs["rgb"]).mean()

        clip_alpha = torch.clamp(outputs["accumulation"],1e-6,1-1e-6)
        beta_loss = torch.log(clip_alpha) + torch.log(1-clip_alpha)
        loss_dict = {"rgb_loss": rgb_loss_fine, 'beta_loss': self.config.beta_loss_scale*beta_loss.mean()}
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb_fine = outputs["rgb"]
        acc_fine = colormaps.apply_colormap(outputs["accumulation"])
        assert self.config.collider_params is not None
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        images_dict = {"img": rgb_fine, "accumulation": acc_fine, "depth": depth_fine}

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)
        assert isinstance(fine_ssim, torch.Tensor)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "psnr": float(fine_psnr),
            "ssim": float(fine_ssim),
            "lpips": float(fine_lpips),
        }
        return metrics_dict, images_dict