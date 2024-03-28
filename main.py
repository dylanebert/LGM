import numpy as np
import rembg
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

from core.models import LGM
from core.multiview import image_to_multiview

model = LGM().half().to("cuda")
ckpt = load_file("pretrained/model_fp16.safetensors", device="cpu")
model.load_state_dict(ckpt, strict=False)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

bg_remover = rembg.new_session()

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

rays_embeddings = model.prepare_default_rays("cuda", elevation=0)
input_image = torch.cat([images, rays_embeddings], dim=1).unsqueeze(0)

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        gaussians = model.forward_gaussians(input_image)

    model.gs.save_ply(gaussians, "output/gaussians.ply")
