"""
Fuzzy Metaballs Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Literal

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import Parameter
from torch.nn import (
    L1Loss,
    MSELoss
)
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.utils.math import safe_normalize



from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler
from nerfstudio.fields.nerfacto_field import NerfactoField

# from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.utils import colormaps

@dataclass
class FMBModelConfig(ModelConfig):
    """Fuzzy Metaballs Model Configuration."""

    _target: Type = field(default_factory=lambda: FMBModel)

    # bg_settings
    background_model: Literal["grid", "mlp", "color", "sh", "hash"] = "mlp"
    far_plane_bg: float = 1000.0
    num_samples_outside: int = 3
    bg_size = 256

    # init settings
    num_gaussians: int = 40
    mean_dist: float = 1e-1
    cov_dist: float = 0.3
    cov_scale: float = 40
    w_scale: float = 5e-2

    # loss settings
    beta_loss_scale: float = 2e-5

    # render settings
    use_two_param: bool = True
    beta1: float = 21.4
    beta2: float = 2.66


def bilinear_interpolate_torch(im, x, y):
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)
    
    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]
    
    wa = ((x1-x) * (y1-y))[:,None]
    wb = ((x1-x) * (y-y0))[:,None]
    wc = ((x-x0) * (y1-y))[:,None]
    wd = ((x-x0) * (y-y0))[:,None]

    return Ia*wa + Ib*wb + Ic*wc + Id*wd


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
        #torch.autograd.set_detect_anomaly(True)
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
        self.precs = Parameter(self.config.cov_dist*(torch.tile(torch.eye(3),(num_g,1,)).reshape((-1,3,3))+ 0.1*torch.randn(num_g,3,3)))
        self.wlogs = Parameter(torch.log(self.config.w_scale*(0.95+0.1*torch.rand(num_g))))
        self.colors = Parameter(torch.randn((num_g,3)))

        # losses
        self.rgb_loss = MSELoss()#L1Loss()

        # renderers
        #self.renderer_rgb = RGBRenderer()#background_color=self.config.background_color)
        #self.renderer_accumulation = AccumulationRenderer()
        #self.renderer_depth = DepthRenderer(method="median")
        #self.renderer_expected_depth = DepthRenderer(method="expected")
        #self.renderer_normals = NormalsRenderer()

        if self.config.background_model == "mlp":
            position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
            )
            direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
            )
            #direction_encoding = SHEncoding()

            self.scene_contraction = SceneContraction(order=float("inf"))
            self.field_background = NeRFField(
                #base_mlp_num_layers = 3, 
                #base_mlp_layer_width = 48, 
                #head_mlp_num_layers = 2, 
                #head_mlp_layer_width = 24,
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
                spatial_distortion=self.scene_contraction,
            )
            self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)
            self.renderer_rgb = RGBRenderer()
        elif self.config.background_model == 'hash':
            self.scene_contraction = SceneContraction(order=float("inf"))
            self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)
            self.renderer_rgb = RGBRenderer()
            self.field_background = NerfactoField(
                self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_average_appearance_embedding=False,
                hidden_dim = 32,
                hidden_dim_color = 32,
                num_layers_color = 2,
                num_levels = 12,
                base_res  = 8,
                max_res = 1024,
                log2_hashmap_size  = 16,

            )
        elif self.config.background_model == "grid":
            self.field_background = Parameter(torch.randn((self.config.bg_size,self.config.bg_size,3)))
        elif self.config.background_model == 'sh':
            self.encoding = SHEncoding()
            self.field_background = Parameter(torch.randn((self.encoding.get_out_dim(),3)))

        else:
            # dummy background model
            self.field_background = Parameter(torch.randn(1,3))


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
        param_groups["colors"] =[self.colors]
        param_groups["background"] = (
            [self.field_background]
            if isinstance(self.field_background, Parameter)
            else list(self.field_background.parameters())
        )
        #param_groups = {'fields':sum(param_groups.values(),[])}
        return param_groups
 
    #@torch.compile()
    def get_outputs(self, ray_bundle: RayBundle):
        if self.means is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        tprec = self.config.cov_scale*torch.tril(self.precs)

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
        norm_b_est = safe_normalize(projp2)

        norm_sign = -torch.sign(torch.einsum('bki,bi->bk',norm_b_est,tr))
        norm_est = norm_sign[:,:,None] * norm_b_est

        # combine
        sig1 = (z > 0) + 1e-9
        d2_z = torch.nan_to_num(d2/sig1)

        if self.config.use_two_param:
            est_alpha = 1-torch.exp(-(sig1*torch.exp(d2)).sum(axis=1))
            # trying to get this numerically stable
            est_logit = -torch.where(sig1>0.5,z,-z)*self.config.beta1 + self.config.beta2*d2_z
            est_logit = est_logit - est_logit.max(dim=1,keepdims=True)[0]
            w = torch.exp(est_logit)
            #w = sig1*torch.softmax(est_logit,1)+1e-46
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
        final_norm = safe_normalize(final_norm)

        pad_alpha = est_alpha[:,None]
        obj_color = w @ torch.sigmoid(self.colors)
        if self.config.background_model == "mlp" or self.config.background_model == "hash":
            ray_bundle.nears = ray_bundle.fars
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            field_outputs_bg = self.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            bg_colors = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
        elif self.config.background_model == 'grid':
            azimuth = torch.arctan2(tr[:, 1], tr[:, 0])
            elevation = torch.arctan2(tr[:, 2], torch.sqrt(tr[:, 0]**2 + tr[:, 1]**2))
            idx1, idx2 = (azimuth+torch.pi)/(2*torch.pi),(elevation+0.5*torch.pi)/torch.pi
            BG_S = self.config.bg_size
            #c_look =  self.field_background[(idx1*(BG_S-1)).int(),(idx2*(BG_S-1)).int()]
            c_look = bilinear_interpolate_torch(self.field_background,idx1*(BG_S-1),idx2*(BG_S-1))
            bg_colors = torch.sigmoid(c_look)
        elif self.config.background_model == 'sh' or self.config.background_model == 'hash':
            product = (self.encoding(tr)[:,:,None] * self.field_background[None]).sum(axis=1)
            bg_colors = torch.sigmoid(product)

        else:
            bg_colors = torch.sigmoid(self.field_background)

        final_color = (1-pad_alpha) * bg_colors + pad_alpha * obj_color
        final_z = (1-est_alpha) * self.config.far_plane_bg + est_alpha * final_z
        final_norm = (1-pad_alpha) * (-tr) + pad_alpha * final_norm
        final_norm = safe_normalize(final_norm)
        
        # green bg for foreground
        fg_bg_c = torch.zeros((1,3),device=obj_color.device) 
        fg_bg_c[0][1] = 1.0

        fg_color = (1-pad_alpha) * fg_bg_c + pad_alpha * obj_color
        outputs = {
            "rgb": final_color,
            "rgb_fg": fg_color,
            "rgb_bg": fg_color,
            "accumulation": est_alpha,
            "depth": final_z[:,None],
            "normals": 0.5+0.5*final_norm, # strange? But exporters wants [0,1]
        }
        if self.config.background_model != "color":
            outputs['rgb_bg'] = bg_colors
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)
        #pred_img, gt_img = self.renderer_rgb.blend_background_for_loss_computation(image,outputs["rgb"],outputs["accumulation"])
        #rgb_loss_fine = self.rgb_loss(pred_img, gt_img)#.mean()
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb"])#.mean()

        clip_alpha = torch.clamp(outputs["accumulation"],0.01,0.99)
        beta_loss = torch.log(clip_alpha) + torch.log(1-clip_alpha)
        #if torch.isnan(rgb_loss_fine).item():
        #    raise ValueError("NaN Loss")
        loss_dict = {"rgb_loss": rgb_loss_fine, 'beta_loss': self.config.beta_loss_scale*beta_loss.mean()}
        #print(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        #image = self.renderer_rgb.blend_background(image)
        rgb_fine = outputs["rgb"]
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb"])#.mean()

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
            "psnr": float(fine_psnr),
            "ssim": float(fine_ssim),
            "lpips": float(fine_lpips),
            "rgb": float(rgb_loss_fine),
        }
        return metrics_dict, images_dict