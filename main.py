from diffusers import DiffusionPipeline

from multiview import image_to_multiview

pipeline = DiffusionPipeline.from_pretrained(
    "dylanebert/LGM",
    custom_pipeline="dylanebert/LGM",
    trust_remote_code=True,
)

images = image_to_multiview(
    "https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/cat_statue.jpg",
    "",
    True,
)

gaussians = pipeline(images)
pipeline.save_ply(gaussians, "output/gaussians.ply")
