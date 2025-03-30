#%%
from copy import copy
from diffusers import FluxPipeline
from datasets import Dataset
from pathlib import Path
import numpy as np
import orjson
import torch
# %%
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, device_map="balanced")
#%%
from tqdm.auto import tqdm
import contextlib
class FluxDataset(torch.utils.data.Dataset):
    def __init__(self, pipe, directory="dataset", all_images=None, limit_batch_size = None):
        if all_images is not None:
            self.all_images = all_images
        else:
            self.all_images = []
            for json_file in tqdm(list(Path(directory).glob("*/*.json"))):
                batch_n = int(json_file.stem.rpartition("-")[-1])
                npz_path = json_file.with_suffix(".npz")
                data = orjson.loads(json_file.read_text())
                prompts = data["prompts"]
                external_keys = {k: data[k] for k in data
                    if k not in ("prompts", "batch_size")}
                if limit_batch_size is not None:
                    prompts = prompts[:limit_batch_size]
                self.all_images.append(dict(
                    prompts=prompts, npz_path=str(npz_path),
                    batch_n=batch_n, idxs=list(range(len(prompts))), **external_keys,
                ))
        self.nf4 = np.asarray([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ])
        self.pipe = pipe
        self.pipe_type = "schnell" if "schnell" in pipe.config._name_or_path else "dev"
        self.ae = pipe.vae
    def __len__(self):
        return len(self.all_images)
    def latent_to_img(self, latent, decode=False):
        """Convert latent representation to image"""
        latent = np.stack((latent & 0x0F, (latent & 0xF0) >> 4), -1).reshape(
            *latent.shape[:-1], -1
        )
        latent = self.nf4[latent]
        latent = latent * 6.0
        
        if not decode:
            return latent

        image = self.ae.decode(
            z=torch.from_numpy(
                latent / self.ae.config.scaling_factor + self.ae.config.shift_factor
            ).to(torch.bfloat16)
        ).sample
        image_np = (
            ((image + 1) * 127)
            .clip(0, 255)
            .to(torch.uint8)
            .numpy()
            .squeeze()
            .transpose(1, 2, 0)
        )

        return image_np
    @torch.inference_mode()
    def __getitem__(self, idx):
        data = self.all_images[idx]
        height, width, guidance_scale = data["height"], data["width"], data["guidance_scale"]
        prompts = data["prompts"]
        batch_size = len(prompts)
        idxs = data["idxs"]
        compressed = np.load(data["npz_path"])["arr_0"]
        compressed = compressed[idxs]
        uncompressed = self.latent_to_img(compressed)
        latents = torch.from_numpy(uncompressed).to(torch.bfloat16)
        latents = latents.unflatten(-2, (-1, 2)).unflatten(-1, (-1, 2))
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.flatten(1, 2).flatten(-3, -1)
        noise = torch.randn_like(latents)
        guidance = torch.tensor([guidance_scale]).to(latents).expand(latents.shape[0])
        height_ = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_ = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))
        latent_image_ids = self.pipe._prepare_latent_image_ids(
            batch_size, height_ // 2, width_ // 2, latents.device, latents.dtype)
        timesteps = torch.rand(batch_size, device=latents.device, dtype=latents.dtype)
        ts = timesteps[:, None, None]
        mixed = (1 - ts) * latents + ts * noise
        
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=None,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=0.0,
        )
        
        return dict(
            n_tokens=latents.shape[0] * latents.shape[1],
            kwargs=dict(
                hidden_states=mixed,
                timestep=timesteps,
                guidance=None if self.pipe_type != "dev" else guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                # joint_attention_kwargs=self.pipe.joint_attention_kwargs,
                return_dict=False,
            )
        )
    def select(self, indices):
        new_all_images = [self.all_images[i] for i in indices]
        return FluxDataset(self.pipe, all_images=new_all_images)

class RecordDouble(torch.nn.Module):
    def __init__(self, child):
        super().__init__()
        self.child = child
        self.text_output = torch.nn.Identity()
        self.image_output = torch.nn.Identity()
    
    def forward(self, *args, **kwargs):
        text_output, image_output = self.child(*args, **kwargs)
        return self.text_output(text_output), self.image_output(image_output)

class RecordSingle(torch.nn.Module):
    def __init__(self, child):
        super().__init__()
        self.child = child
        self.text_output = torch.nn.Identity()
        self.image_output = torch.nn.Identity()
    
    def forward(self, *args, **kwargs):
        outputs = self.child(*args, **kwargs)
        text_output, image_output = \
            outputs[..., :512, :], outputs[..., 512:, :]
        self.text_output(text_output)
        self.image_output(image_output)
        return outputs

@contextlib.contextmanager
def temporary_transformer_blocks(transformer):
    # Save original blocks
    original_blocks = transformer.transformer_blocks
    original_single_blocks = transformer.single_transformer_blocks

    new_transformer_blocks = torch.nn.ModuleList([
        RecordDouble(block) for block in transformer.transformer_blocks
    ])
    new_single_transformer_blocks = torch.nn.ModuleList([
        RecordSingle(block) for block in transformer.single_transformer_blocks
    ])
    
    try:
        # Replace with new blocks
        transformer.transformer_blocks = new_transformer_blocks
        transformer.single_transformer_blocks = new_single_transformer_blocks
        yield
    finally:
        # Restore original blocks
        transformer.transformer_blocks = original_blocks
        transformer.single_transformer_blocks = original_single_blocks
