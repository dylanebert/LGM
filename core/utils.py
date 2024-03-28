import numpy as np
import torch
import torch.nn.functional as F


def get_rays(pose, h, w, fovy, opengl=True):
    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d)

    rays_o = rays_o.view(h, w, 3)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1).view(h, w, 3)

    return rays_o, rays_d
