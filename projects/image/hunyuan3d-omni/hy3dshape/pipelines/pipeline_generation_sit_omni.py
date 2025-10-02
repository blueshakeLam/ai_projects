# -*- coding: utf-8 -*-
"""
Tencent is pleased to support the open source community by making Tencent Hunyuan 3D Omni available.

Copyright (C) 2025 Tencent.  All rights reserved. The below software and/or models in this 
distribution may have been modified by Tencent ("Tencent Modifications"). All Tencent Modifications 
are Copyright (C) Tencent.

Tencent Hunyuan 3D Omni is licensed under the TENCENT HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT 
except for the third-party components listed below, which is licensed under different terms. 
Tencent Hunyuan 3D Omni does not impose any additional limitations beyond what is outlined in the 
respective licenses of these third-party components. Users must comply with all terms and conditions 
of original licenses of these third-party components and must ensure that the usage of the third party 
components adheres to all relevant laws and regulations. 

For avoidance of doubts, Tencent Hunyuan 3D Omni means training code, inference-enabling code, parameters, 
and/or weights of this Model, which are made publicly available by Tencent in accordance with TENCENT 
HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT.
"""

from typing import List, Optional, Union
import os
import json
import numpy as np
import trimesh
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from huggingface_hub import snapshot_download
from diffusers.utils.torch_utils import randn_tensor

from .utils import init_from_ckpt, export_to_trimesh, synchronize_timer
from hy3dshape.models.utils.misc import get_config_from_file, instantiate_from_config

from hunyuan3d_omni_config import *
class Hunyuan3DOmniSiTFlowMatchingPipeline:
    '''
    This pipeline is designed for generating 3D shapes using the Hunyuan3DOmni model with SiT flow matching.
    '''
    @classmethod
    def from_pretrained(cls,
                        model_path,
                        variant=None,
                        device='cuda',
                        dtype=torch.float16,
                        resume_download=False,
                        force_download=False,
                        revision="main",
                        **kwargs):

        if os.path.exists(model_path):
            print(f'Loading model from local path: {model_path}')
        else:
            repo_id = model_path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, repo_id))
            print(f'Loading model from huggingface cache: {model_path}')
            if not os.path.exists(model_path):
                print(f'Not Found {model_path}')
                print(f'Downloading model from huggingface: {repo_id}')

            path = snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=resume_download,
                force_download=force_download, 
                revision=revision,
            )
            print('path', path)

        ckpt_name = 'pytorch_model.bin'
        submodules = dict()
        for submodule_name in ['model', 'vae','cond_encoder', 'scheduler', 'image_processor']:
            config_path = os.path.join(model_path, submodule_name, 'config.json')
            config = get_config_from_file(config_path)
            if submodule_name == 'cond_encoder':
                config['params']['image_encoder']['params']['version'] = get_facebook_dinov2_large_path()
            if submodule_name == "scheduler":
                if 'transport' in config and 'sampler' in config:
                    transport = instantiate_from_config(config.transport)
                    submodule = instantiate_from_config(config.sampler, transport=transport)
                else:
                    scheduler_config = kwargs.get('scheduler_config', config.scheduler_cfg.get('denoise'))
                    submodule = instantiate_from_config(scheduler_config)
                submodules[submodule_name] = submodule
                print(f'Loaded {submodule_name}')
                continue

            submodule = instantiate_from_config(config)
            if isinstance(submodule, nn.Module):
                ckpt_path = os.path.join(model_path, submodule_name, ckpt_name)
                if variant == 'ema':
                    ckpt_name = 'pytorch_model_ema.bin'
                    ckpt_path = os.path.join(model_path, submodule_name, ckpt_name)
                    if not os.path.exists(ckpt_path):
                        ckpt_name = 'pytorch_model.bin'
                        ckpt_path = os.path.join(model_path, submodule_name, ckpt_name)
                if os.path.exists(ckpt_path):
                    missing, unexpected = submodule.load_state_dict(
                        torch.load(ckpt_path, map_location='cpu'), strict=False)
                    print(f"Loaded {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
                    if len(missing) > 0:
                        print(f"Missing Keys: {Counter([s.split('.')[0] for s in missing])}")
                    if len(unexpected) > 0:
                        print(f"Unexpected Keys: {Counter([s.split('.')[0] for s in unexpected])}")

                submodule.disable_drop = True
                submodule = submodule.to(device=device, dtype=dtype)
                submodule.eval()
            submodules[submodule_name] = submodule
            print(f'Loaded {submodule_name}')

        vae_config_path = os.path.join(model_path, 'vae', 'config.json')
        vae_config = get_config_from_file(vae_config_path)
        model_kwargs = dict(
            scale_factor=vae_config['scale_factor'],
            device=device,
            dtype=dtype,
        )
        model_kwargs.update(submodules)
        model_kwargs.update(kwargs)
        return cls(**model_kwargs)

    def __init__(self,
                vae=None,
                model=None,
                scheduler=None,
                cond_encoder=None,
                image_processor=None,
                scale_factor=1.0,
                device='cuda',
                dtype=torch.float16,
                **kwargs):
        self.vae = vae                           # Shape-VAE
        self.model = model                       # Shape-DiT
        self.scheduler = scheduler               # Denoiser Scheduler
        self.cond_encoder = cond_encoder         # Condition Encoder
        self.image_processor = image_processor   # Image Processor
        self.scale_factor = scale_factor
        self.device = torch.device(device)
        self.dtype = dtype

    def encode_cond(self, image, surface, pose, bbox, point, voxel, do_classifier_free_guidance):
        bsz = image.shape[0]
        cond = self.cond_encoder(image=image, surface=surface, pose=pose, bbox=bbox, point=point, voxel=voxel)
        sampled_point = cond['cond_point']
        if do_classifier_free_guidance:
            img_uncond = self.cond_encoder.unconditional_embedding(bsz)["dino"]["last_hidden_state"]
            uncond = torch.cat([img_uncond, cond['cond'][:, img_uncond.shape[1]:, :]], dim=1)
            un_cond = {'cond': uncond}
            cond = {'cond': torch.cat((cond['cond'], un_cond['cond']), dim=0)}
        return cond, sampled_point

    def prepare_latents(self, timestep, batch_size, dtype, device, generator, latents=None):
        shape = (batch_size, *self.vae.latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents


    def prepare_image(self, image, mask):
        if isinstance(image, torch.Tensor):
            return image, mask

        if isinstance(image, str):
            if not os.path.exists(image):
                raise ValueError(f"Image path {image} does not exist.")
            image = [image]

        image_pts, mask_pts = [], []
        for img in image:
            image_pt, mask_pt = self.image_processor(img, return_mask=True, return_view_idx=False)
            image_pts.append(image_pt)
            mask_pts.append(mask_pt)

        image_pts = torch.cat(image_pts, dim=0).to(self.device, dtype=self.dtype)
        if mask_pts[0] is not None:
            mask_pts = torch.cat(mask_pts, dim=0).to(self.device, dtype=self.dtype)
        else:
            mask_pts = None
        return image_pts, mask_pts

    @torch.no_grad()
    def __call__(
        self,
        image,
        mask: torch.Tensor = None,
        surface: Union[str, List[str], torch.Tensor] = None,
        pose: Union[str, List[str], torch.Tensor] = None,
        bbox: Union[str, List[str], torch.Tensor] = None,
        point: Union[str, List[str], torch.Tensor] = None,
        voxel: Union[str, List[str], torch.Tensor] = None,
        fast_decode: bool = False,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        prev_guidance_scale: float = None,
        guidance_scale: float = 7.5,
        dino_image_size: int = None,
        generator=None,
        box_v=1.01,
        octree_depth=8,
        octree_resolution=None,
        mc_level=0.0,
        mc_mode='mc',
        num_chunks=8000,
        sigmoid=False,
        output_type: Optional[str] = "trimesh",
        sampling_method='euler',
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        '''
            Generate 3D shapes using the Hunyuan3DOmni model with SiT flow matching.
            Args:
                image: the input image.
                mask: the input mask.
                surface: the input surface.
                pose: the input pose.
                bbox: the input bbox.
                point: the input point.
                voxel: the input voxel.
                fast_decode: whether to use fast decode (flashvdm).
                prompt: the input prompt.
                num_inference_steps: the number of inference steps.
                timesteps: the timesteps.
                prev_guidance_scale: the previous guidance scale.
                guidance_scale: the guidance scale.
                dino_image_size: the dino image size.
                generator: the generator.
                box_v: the box v.
                octree_depth: the octree depth.
                octree_resolution: the octree resolution.
                mc_level: the mc level.
                mc_mode: the mc mode.
                num_chunks: the number of chunks.
                sigmoid: whether to use sigmoid.
                output_type: the output type.
        '''

        self.vae.fast_decode = fast_decode

        if dino_image_size is not None and hasattr(self.cond_encoder, 'setup'):
            self.cond_encoder.setup(image_size=dino_image_size)

        device, dtype = self.device, self.dtype

        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'params') and self.model.params.guidance_embed is True)

        image, mask = self.prepare_image(image, mask)
        cond, sampled_point = self.encode_cond(image, surface, pose, bbox, point, voxel, do_classifier_free_guidance)
        batch_size = image.shape[0]

        latents = self.prepare_latents(None, batch_size, dtype, device, generator)
        sample_fn = self.scheduler.sample_ode(num_steps=num_inference_steps, sampling_method=sampling_method)
        guidance = None
        if hasattr(self.model, 'params') and self.model.params.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
            print("[Guidance Distilled Model] Using guidance embeddings.")

        def denoise_with_cfg(inputs, ts, contexts):
            noise_pred = self.model(inputs, ts, contexts, guidance=guidance)
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                return torch.cat([noise_pred, noise_pred], dim=0)
            return noise_pred

        if do_classifier_free_guidance:
            latents = torch.cat([latents, latents])

        latents = sample_fn(latents, denoise_with_cfg, contexts=cond)[-1]
        if do_classifier_free_guidance:
            latents = latents.chunk(2, dim=0)[0]

        if not output_type == "latent":
            latents = 1. / self.scale_factor * latents
            print("converting latents dtype to ", next(self.vae.parameters()).dtype)
            latents = latents.to(next(self.vae.parameters()).dtype)

            shapes = self.vae.decode(
                latents,
                octree_depth=octree_depth,
                bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_mode=mc_mode,
                sigmoid=sigmoid,
            )
        else:
            shapes = latents

        if output_type == 'trimesh':
            shapes = export_to_trimesh(shapes)
        return {'shapes':[shapes], 'sampled_point':sampled_point, 'image': image}
