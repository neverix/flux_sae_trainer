#%%
%load_ext autoreload
%autoreload 2
from image_data import temporary_transformer_blocks, FluxDataset
from diffusers import FluxPipeline
from sparsify import TrainConfig, SaeConfig, Trainer
import torch
import fire
#%%
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device_map="balanced")
#%%
bs = 4
dataset = FluxDataset(pipe, directory="dataset", limit_batch_size=bs)
#%%
config = TrainConfig(
    sae=SaeConfig(expansion_factor=8),
    batch_size=1,
    hookpoints=["transformer_blocks.16.image_output"],
)
# %%
pipe.transformer.dummy_inputs = next(iter(dataset))["kwargs"]
#%%
with temporary_transformer_blocks(pipe.transformer):
    trainer = Trainer(config, dataset, pipe.transformer)
    trainer.fit()
#%%
# def main():
#     pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, device_map="balanced")
#     # pipe_dev = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device_map="balanced")


# if __name__ == "__main__":
#     fire.Fire(main)

pipe.transformer.transformer_blocks[16].image_output