
# https://github.com/ashawkey/stable-dreamfusion/issues/96
# Question: Generate image using SDS

import math
from tqdm import tqdm
import torch
import torch.nn as nn
from nerf.sd import StableDiffusion, seed_everything
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


device = 'cuda:0'
guidance = StableDiffusion(device)
guidance.vae.encoder = None

prompt = 'a photorealistic hamburger'
text_embeddings = guidance.get_text_embeds(prompt, '')
guidance.text_encoder.to('cpu')
torch.cuda.empty_cache()

seed_everything(42)
latents = nn.Parameter(torch.randn(1, 4, 64, 64, device=device))
optimizer = torch.optim.AdamW([latents], lr=1e-1, weight_decay=0)
num_steps = 1000
scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(num_steps*1.5))

for step in tqdm(range(num_steps)):
    optimizer.zero_grad()

    t = torch.randint(guidance.min_step, guidance.max_step + 1, [1], dtype=torch.long, device=guidance.device)
    with torch.no_grad():
        # add noise
        noise = torch.randn_like(latents)
        latents_noisy = guidance.scheduler.add_noise(latents, noise, t)
        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2)
        noise_pred = guidance.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance (high scale from paper!)
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + 100 * (noise_pred_text - noise_pred_uncond)

    w = (1 - guidance.alphas[t])
    grad = w * (noise_pred - noise)

    latents.backward(gradient=grad, retain_graph=True)

    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        # Static threshold
        latents.data = latents.data.clip(-1, 1)
        # Dynamic thresholding
        #s = torch.as_tensor(np.percentile(latents.abs().cpu().numpy(), 90, axis=(1,2,3)), dtype=latents.dtype).to(device)
        #latents.data = latents.clip(-s, s) / s

    if step > 0 and step % 100 == 0:
        rgb = guidance.decode_latents(latents)
        img = rgb.detach().squeeze(0).permute(1,2,0).cpu().numpy()
        print('[INFO] save image', img.shape, img.min(), img.max())
        plt.imsave(f'tmp_lat_img_{step}.jpg', img)