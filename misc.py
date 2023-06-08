
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler,DDIMScheduler
import torch

def base_sd():
    model_id = "stabilityai/stable-diffusion-2-base"

    # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "realistic photography of a cat"
    image = pipe(prompt).images[0]

    image.save("cat——DDIMScheduler.png")

def ControlNet():
    # !pip install opencv-python transformers accelerate
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from diffusers.utils import load_image
    import numpy as np
    import torch

    import cv2
    from PIL import Image

    # download an image
    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )
    image = np.array(image)

    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    # generate image
    generator = torch.manual_seed(0)
    image = pipe(
        "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
    ).images[0]
    image.save("futuristic-looking_woman.png")

def ControlNet_pose2img():
    # ! pip    install "git+https://github.com/huggingface/diffusers.git"  # Diffusers latest version (in development)
    # ! pip    install    transformers    accelerate    safetensors  xformers==0.0.19   opencv-python
    # torch2.0 xformers==0.0.19
    # ! pip install controlnet_hinter==0.0.5  # image preprocess library for controlnet in development version

    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers.utils import load_image
    import torch
    import numpy as np
    from PIL import Image
    import cv2

    controlnet = None
    pipe = None
    prompt = "best quality, extremely detailed, football, a boy"
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None
    ).to('cuda')
    pipe.enable_xformers_memory_efficient_attention()

    pose_image = load_image('https://huggingface.co/takuma104/controlnet_dev/resolve/main/pose.png')

    generator = torch.Generator(device="cpu").manual_seed(0)
    image = pipe(prompt=prompt,
                 negative_prompt="lowres, bad anatomy, worst quality, low quality",
                 image=pose_image,
                 generator=generator,
                 num_inference_steps=30).images[0]

    image.save(prompt + ".png")

def ControlNet_pose2img2():
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers.utils import load_image
    import torch
    import numpy as np
    from PIL import Image
    import cv2
    import controlnet_hinter
    original_image = load_image(
        "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_imgvar/input_image_vermeer.png")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None
    ).to('cuda')
    control_image = controlnet_hinter.hint_openpose(original_image)
    generator = torch.Generator(device="cuda").manual_seed(0)
    image = pipe(prompt="best quality, extremely detailed",
                 negative_prompt="lowres, bad anatomy, worst quality, low quality",
                 image=control_image,
                 generator=generator,
                 num_inference_steps=30).images[0]


def inpaint_mask2img():
    import PIL
    import requests
    import torch
    from io import BytesIO

    from diffusers import StableDiffusionInpaintPipeline

    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    init_image = download_image(img_url).resize((512, 512))
    mask_image = download_image(mask_url).resize((512, 512))

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]


if __name__ == '__main__':
    ControlNet_pose2img()
