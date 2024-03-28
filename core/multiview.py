import numpy as np
from gradio_client import Client, file
from PIL import Image


def text_to_multiview(prompt, split=False):
    client = Client("dylanebert/multi_view_diffusion")
    result_path = client.predict(prompt, api_name="/text_to_mv")
    result = np.array(Image.open(result_path), dtype=np.uint8)
    if split:
        vertical_split = np.split(result, 2, axis=0)
        result = np.vstack([np.split(part, 2, axis=1) for part in vertical_split])
    return result


def image_to_multiview(image_path, prompt="", split=False):
    client = Client("dylanebert/multi_view_diffusion")
    image = file(image_path)
    result_path = client.predict(image, prompt, api_name="/image_to_mv")
    result = np.array(Image.open(result_path), dtype=np.uint8)
    if split:
        vertical_split = np.split(result, 2, axis=0)
        result = np.vstack([np.split(part, 2, axis=1) for part in vertical_split])
    return result


if __name__ == "__main__":
    text_to_mv_result = text_to_multiview("a cat statue")
    image_to_mv_result = image_to_multiview(
        "https://huggingface.co/datasets/dylanebert/3d-arena/resolve/main/cat_statue.jpg"
    )
    Image.fromarray(text_to_mv_result).save("output/text_to_multiview_result.jpg")
    Image.fromarray(image_to_mv_result).save("output/image_to_multiview_result.jpg")
