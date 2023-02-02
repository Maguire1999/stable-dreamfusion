
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler,DDIMScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-base"

# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "realistic photography of a cat"
image = pipe(prompt).images[0]

image.save("cat——DDIMScheduler.png")
