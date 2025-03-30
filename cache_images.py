#%%
import sys
n_gpus = 6
gpu_id = int(sys.argv[1])
#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# from collections import OrderedDict
import numpy as np
from datasets import load_dataset
from more_itertools import chunked
from tqdm.auto import tqdm
import numpy, torch
import json, os
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, device_map="balanced")

dataset_dir = f"dataset/{gpu_id}"
os.makedirs(dataset_dir, exist_ok=True)

torch.set_grad_enabled(False)
#%%
image_max = 6.0
guidance_scale = 3.5
num_inference_steps = 4  # also input to the algorithm (1 or 4)
batch_size = 128

# 512 or 256, same as height, input to the algorithm (possible config)
width = 512
height = 512
d_model = 3072

nf4 = np.asarray(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)

@torch.compile(mode="max-autotune")
def strip_high_cossim(x, threshold=0.8, remove_pct=0.5):
    x = x - x.mean(0)
    x = x / x.norm(dim=-1, keepdim=True)
    has_duplicate = (
        (x @ x.T - 2 * torch.eye(x.shape[0], device=x.device, dtype=x.dtype)) > threshold).any(dim=-1)
    rand_mask = (torch.rand_like(
        has_duplicate, dtype=torch.float16) < remove_pct)
    return ~(has_duplicate & rand_mask)

prompts_dataset = load_dataset("opendiffusionai/cc12m-cleaned")
prompts_iterator = prompts_dataset["train"]["caption_llava_short"]
prompts_iterator = prompts_iterator[:len(prompts_iterator) // 4]
#%%
starts = np.linspace(0, len(prompts_iterator), n_gpus + 1)
starts = np.floor(starts).astype(int).tolist()
chunks = [prompts_iterator[start:end] for start, end in zip(starts, starts[1:])]
start = starts[gpu_id]
prompts_iterator = chunks[gpu_id]
#%%
device = torch.device("cuda:0")
for i, prompts in enumerate(chunked((bar := tqdm(prompts_iterator)), batch_size)):
    json_path = f"{dataset_dir}/batch-{i}.json"
    if os.path.exists(json_path):
        continue
    
    with torch.inference_mode():
        def callback_on_step_end(self, i, t, kwargs):
            global timestep
            timestep = i
            return {}
        pipe.set_progress_bar_config(disable=True)
        with torch.inference_mode():
            latents = pipe(
                prompts,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0),
                return_dict=False,
                callback_on_step_end=callback_on_step_end,
                output_type="latent",
            )[0]
        latents_reshaped = pipe._unpack_latents(
            latents, height, width, pipe.vae_scale_factor)
        latents_to_be_compressed = latents_reshaped.cpu().float().numpy()
        latents_to_save = (latents_to_be_compressed /
                            image_max).clip(-1, 1)
        latents_to_save = np.abs(
            latents_to_save[..., None] - nf4).argmin(-1).astype(np.uint8)
        latents_to_save = (
            (latents_to_save[..., ::2] & 0x0F)
            | ((latents_to_save[..., 1::2] << 4) & 0xF0))
        np.savez_compressed(f"{dataset_dir}/batch-{i}.npz", latents_to_save)
        json.dump(dict(
            prompts=prompts,
            step=i,
            batch_size=batch_size,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ), open(json_path, "w"))
# %%
