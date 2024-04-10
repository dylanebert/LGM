import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import DiffusionPipeline

from multiview import image_to_multiview

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


images = image_to_multiview(
    "https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/cat_statue.jpg",
    "",
    True,
)


images = np.stack([images[1], images[2], images[3], images[0]], axis=0)
images = images.astype(np.float32) / 255.0
images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to("cuda")
images = F.interpolate(
    images,
    size=(256, 256),
    mode="bilinear",
    align_corners=False,
)
images = TF.normalize(images, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/LGM",
    custom_pipeline="dylanebert/LGM",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")

rays_embeddings = pipeline.prepare_default_rays("cuda", elevation=0)
inputs = torch.cat([images, rays_embeddings], dim=1).unsqueeze(0)

with torch.autocast(device_type="cuda", dtype=torch.float16):
    result = pipeline(inputs)

pipeline.gs.save_ply(result, "output/gaussians.ply")
