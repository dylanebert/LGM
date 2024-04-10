from safetensors.torch import load_file

from lgm import LGM, LGMPipeline
from multiview import image_to_multiview

model = LGM()
ckpt = load_file("pretrained/model_fp16.safetensors", device="cpu")
model.load_state_dict(ckpt, strict=False)
pipeline = LGMPipeline(model)

images = image_to_multiview(
    "https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/cat_statue.jpg",
    "",
    True,
)

gaussians = pipeline(images)
pipeline.save_ply(gaussians, "output/gaussians.ply")
