import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.gs import GaussianRenderer
from core.unet import UNet
from core.utils import get_rays


class LGM(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = 256
        self.splat_size = 128
        self.output_size = 512
        self.radius = 1.5
        self.fovy = 49.1

        self.unet = UNet(
            9,
            14,
            down_channels=(64, 128, 256, 512, 1024, 1024),
            down_attention=(False, False, False, True, True, True),
            mid_attention=True,
            up_channels=(1024, 1024, 512, 256, 128),
            up_attention=(True, True, True, False, False),
        )

        self.conv = nn.Conv2d(14, 14, kernel_size=1)
        self.gs = GaussianRenderer(self.fovy, self.output_size)

        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5

    def prepare_default_rays(self, device, elevation=0):
        from kiui.cam import orbit_camera

        cam_poses = np.stack(
            [
                orbit_camera(elevation, 0, radius=self.radius),
                orbit_camera(elevation, 90, radius=self.radius),
                orbit_camera(elevation, 180, radius=self.radius),
                orbit_camera(elevation, 270, radius=self.radius),
            ],
            axis=0,
        )
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(
                cam_poses[i], self.input_size, self.input_size, self.fovy
            )
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1
            )
            rays_embeddings.append(rays_plucker)

        rays_embeddings = (
            torch.stack(rays_embeddings, dim=0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(device)
        )

        return rays_embeddings

    def forward_gaussians(self, images):
        B, V, C, H, W = images.shape
        images = images.view(B * V, C, H, W)

        x = self.unet(images)
        x = self.conv(x)

        x = x.reshape(B, 4, 14, self.splat_size, self.splat_size)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)

        pos = self.pos_act(x[..., 0:3])
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)

        return gaussians

    def forward(self, data):
        results = {}

        images = data["input"]
        gaussians = self.forward_gaussians(images)
        results["gaussians"] = gaussians

        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        results = self.gs.render(
            gaussians,
            data["cam_view"],
            data["cam_view_proj"],
            data["cam_pos"],
            bg_color=bg_color,
        )
        pred_images = results["image"]
        pred_alphas = results["alpha"]

        results["images_pred"] = pred_images
        results["alphas_pred"] = pred_alphas

        return results
