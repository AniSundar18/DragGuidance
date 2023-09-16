from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, ViTModel
from diffusers import StableDiffusionImageVariationPipeline
# suppress partial model loading warning
logging.set_verbosity_error()
import torch.nn.functional as F
import os
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from pnp_utils import *
from energy_function import E_movement, style_loss
import torchvision.transforms as T

def get_module(input_string):
    split_string = input_string.split('.')
    converted_string = ''
    for element in split_string:
        if element.isdigit():
            converted_string += f"[{element}]"
        else:
            converted_string += f".{element}"
    converted_string = converted_string[1:]
    return converted_string

class ReplaceKV(nn.Module):
    def __init__(self, replacement_tensor):
        super(ReplaceKV, self).__init__()
        self.replacement_tensor = replacement_tensor.to('cuda')

    def forward(self, x):
        # Replace the result of fc1 with the replacement tensor
        return self.replacement_tensor

def apply_transform(image):
    transform = T.Compose([
                    T.Resize(
                        (224, 224),
                        interpolation=T.InterpolationMode.BICUBIC,
                        antialias=False,
                        ),
                    T.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]),
                ])
    tensor_image = transform(image).to('cuda')
    return tensor_image

class Preprocess(nn.Module):
    def __init__(self, opt, hf_key=None, scheduler = None):
        super().__init__()
        
        if opt.use_mutual_self_attention:
            self.positions = [
                     'up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k'
                    ,'up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v'
                    ,'up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k'
                    ,'up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v'
                    ,'up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k'
                    ,'up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v'

                    ,'up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k'
                    ,'up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v'
                    ,'up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k'
                    ,'up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v'
                    ,'up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k'
                    ,'up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v'

                    ,'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k'
                    ,'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v'
                    ,'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k'
                    ,'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v'
                    ,'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k'
                    ,'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v'

                    ]
        self.device = opt.device
        self.guidance = opt.guidance
        self.sd_version = opt.sd_version
        self.use_depth = False
        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
            self.use_depth = True
        elif self.sd_version == '1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
        elif self.sd_version == 'variations':
            model_key = "lambdalabs/sd-image-variations-diffusers"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        
        if opt.use_texture:
            self.dino = ViTModel.from_pretrained('facebook/dino-vitb8').to(self.device )

        # Create model
        if opt.condition_type == 'image':
            self.sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
                        "lambdalabs/sd-image-variations-diffusers",
                        revision="v2.0",
                        ).to(self.device)
            self.unet = self.sd_pipe.unet.to(self.device)
            self.vae = self.sd_pipe.vae.to(self.device)
        
        elif opt.condition_type == 'text':
            self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
                                                            torch_dtype=torch.float16).to(self.device)
        
            self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
                                                         torch_dtype=torch.float16).to(self.device)
            self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", revision="fp16",
                                                 torch_dtype=torch.float16).to(self.device)
        
        #layers = get_layers_with_forward_hook(self.unet)

        #for name, _ in layers:
        #    print(name)
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(num_inference_steps=opt.steps + 1)
        print(f'[INFO] loaded stable diffusion!')

        self.inversion_func = self.ddim_inversion

    def sag_masking(self, original_latents, attn_map, map_size, t, eps):
        # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = self.unet.config.attention_head_dim
        if isinstance(h, list):
            h = h[-1]

        # Produce attention mask
        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
        attn_mask = (
            attn_mask.reshape(b, map_size[0], map_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # Blur according to the self-attention mask
        degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)

        # Noise it again to match the noise level
        degraded_latents = self.scheduler.add_noise(degraded_latents, noise=eps, timesteps=t)

        return degraded_latents

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    #@torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=False,
                                timesteps_to_save=None, cyclic_inversion_steps = None):
        F_tgt = None
        timesteps = reversed(self.scheduler.timesteps)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i, t in enumerate(tqdm(timesteps)):
                if cyclic_inversion_steps is not None:
                    if i not in range(len(timesteps) - cyclic_inversion_steps['end'], len(timesteps) - cyclic_inversion_steps['start']):
                        continue
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                if i == 0:
                    eps, F_tgt = self.unet(latent, t, encoder_hidden_states=cond_batch)
                else:
                    eps, _ = self.unet(latent, t, encoder_hidden_states=cond_batch)
                eps = eps.sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if save_latents:
                    torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
                    torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return latent, F_tgt
    
    def ddim_cyclic_sample(self, x, cond, opt, save_latents=False, timesteps_to_save=None, activation = None, F_tgt = None):
        save_path = opt.save_dir
        split = torch.split(x, 1)
        x_guid = split[0]
        features_gen = {}
        features_guid = {}
        #Store the original layers in case of mutual attention
        if opt.use_mutual_self_attention:
            print('Entry')
            unet_layers = {}
            for block in self.positions:
                layer = "self.unet." + get_module(block)
                exec("unet_layers[block] = " + layer)
        if opt.use_texture:
            opt.target_texture = opt.target_texture.detach()
        x_gen = split[1]
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # First few timesteps: Conventional Denoising
            for i, t in enumerate(tqdm(timesteps[:opt.start])):
                    cond_batch = cond
                    # Setting up diffusion parameters
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                    x_gen.requires_grad_(True)
                    x_gen.retain_grad()
                    x_guid = x_guid.detach()
                    cond_batch = cond_batch.detach()
                    
                    #Denoising
                    eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                    eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)
                    eps_gen = eps_gen.sample
                    eps_guid = eps_guid.sample

                    pred_x0_guid = (x_guid - sigma * eps_guid) / mu
                    pred_x0_gen = (x_gen - sigma * eps_gen) / mu
                    x_guid = mu_prev * pred_x0_guid + sigma_prev * eps_guid
                    x_gen = mu_prev * pred_x0_gen + sigma_prev * eps_gen

                    x_gen = x_gen.detach()
            #Self Guidance for given number of cycles
            print('Entering the Cycle')
            #opt.use_mutual_self_attention = False
            count = 0
            for idx in range(opt.num_cycles):
                for i, t in enumerate(tqdm(timesteps[opt.start:opt.end])):
                    count += 1
                    i = i + opt.start
                    cond_batch = cond
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                    
                    
                    x_gen.requires_grad_(True)
                    x_gen.retain_grad()
                    x_guid = x_guid.detach()
                    cond_batch = cond_batch.detach()
                    #eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                    
                    if opt.use_mutual_self_attention and i%opt.num_skips == 0:
                        # Case of Mutual Self Attention
                        eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                        # Obtain Keys and Values as needed
                        features = {}
                        for block in self.positions:
                            name = block
                            features[name] = activation[name].detach().cpu()
                        
                        # Temporarily switch neural network layers
                        for layer in self.positions:
                            command = "self.unet." + get_module(layer) + ' = ReplaceKV(replacement_tensor = features[layer])' 
                            exec(command)
                        eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)
                    else:
                        eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                        eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)

                    eps_gen = eps_gen.sample
                    eps_guid = eps_guid.sample

                    E = E_movement(F_guid = F_guid, F_gen = F_gen, opt = opt, timestep = i)
                    if E != 0:
                        E.backward()
                        eps_gen = eps_gen + opt.scale * sigma * x_gen.grad

                    pred_x0_guid = (x_guid - sigma * eps_guid) / mu
                    pred_x0_gen = (x_gen - sigma * eps_gen) / mu
                    x_guid = mu_prev * pred_x0_guid + sigma_prev * eps_guid
                    x_gen = mu_prev * pred_x0_gen + sigma_prev * eps_gen
                    #Save Feats for Tracking
                    if opt.save_feats and count%4 == 0:
                        features_gen[str(count)] = F_gen['2'].detach().to('cpu')
                        features_guid[str(count)] = F_guid['2'].detach().to('cpu')
                    #Replace the layers for the guidance branch
                    if opt.use_mutual_self_attention and i%opt.num_skips == 0:
                        for layer in self.positions:
                            command = "self.unet." + get_module(layer) + ' = unet_layers[layer]' 
                            exec(command)
                    
                    x_gen = x_gen.detach()
                # Invert it back to the start timestep
                if idx < opt.num_cycles - 1:
                    print('Invert ' + str(idx))
                    inverted_x, _ = self.inversion_func(cond, torch.cat((x_gen, x_guid)), opt.save_path, cyclic_inversion_steps={'start':opt.start, 'end':opt.end})
                    
                else:
                    inverted_x, _ = self.inversion_func(cond, torch.cat((x_gen, x_guid)), opt.save_path, cyclic_inversion_steps={'start':opt.start, 'end':opt.end})
                    x_gen = inverted_x[0]
                    x_guid = inverted_x[1]
                    x_gen = torch.unsqueeze(x_gen, 0)
                    x_guid = torch.unsqueeze(x_guid, 0)
                    x_gen.detach()
                    x_gen = x_gen.to(torch.float16) 
                    x_guid = x_guid.to(torch.float16) 
            print('Exiting the Cycle')
            #Final Set
            for i, t in enumerate(tqdm(timesteps[opt.start:])):
                    i = i + opt.start
                    cond_batch = cond
                    # Setting up diffusion parameters
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5


                    if opt.isStochastic and i >= 350:
                        sigma_t = (((1 - alpha_prod_t_prev)/(1 - alpha_prod_t))**0.5)*(1 - alpha_prod_t/alpha_prod_t_prev)**0.5
                        sigma_prev = (1 - alpha_prod_t_prev - sigma_t**2) ** 0.5
                    else:
                        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5

                    x_gen.requires_grad_(True)
                    x_gen.retain_grad()
                    x_guid = x_guid.detach()
                    cond_batch = cond_batch.detach()
                    
                    #Denoising
                    if opt.use_mutual_self_attention:
                        # Case of Mutual Self Attention
                        eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                        # Obtain Keys and Values as needed
                        features = {}
                        for block in self.positions:
                            name = block
                            features[name] = activation[name].detach().cpu()
                        
                        # Temporarily switch neural network layers
                        for layer in self.positions:
                            command = "self.unet." + get_module(layer) + ' = ReplaceKV(replacement_tensor = features[layer])' 
                            exec(command)
                        eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)
                    else:
                        eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                        eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)
                    eps_gen = eps_gen.sample
                    eps_guid = eps_guid.sample

                    pred_x0_guid = (x_guid - sigma * eps_guid) / mu
                    pred_x0_gen = (x_gen - sigma * eps_gen) / mu
                    x_guid = mu_prev * pred_x0_guid + sigma_prev * eps_guid
                    x_gen = mu_prev * pred_x0_gen + sigma_prev * eps_gen
                    if opt.isStochastic and i >= 350:
                        x_gen  = x_gen + sigma_t * torch.randn_like(x_gen)
                    if opt.use_mutual_self_attention:
                        for layer in self.positions:
                            command = "self.unet." + get_module(layer) + ' = unet_layers[layer]' 
                            exec(command)
                    x_gen = x_gen.detach()

            x = torch.cat((x_guid, x_gen), dim = 0)
            if save_latents:
                torch.save(x, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return x, features_gen, features_guid

    
    def ddim_cyclic_sag_sample(self, x, cond, opt, save_latents=False, timesteps_to_save=None, activation = None, F_tgt = None):
        save_path = opt.save_dir
        split = torch.split(x, 1)
        x_guid = split[0]
        features_gen = {}
        features_guid = {}
        #Store the original layers in case of mutual attention
        if opt.use_mutual_self_attention:
            print('Entry')
            unet_layers = {}
            for block in self.positions:
                layer = "self.unet." + get_module(block)
                exec("unet_layers[block] = " + layer)
        if opt.use_texture:
            opt.target_texture = opt.target_texture.detach()
        x_gen = split[1]
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # First few timesteps: Conventional Denoising
            for i, t in enumerate(tqdm(timesteps[:opt.start])):
                    cond_batch = cond
                    # Setting up diffusion parameters
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                    x_gen.requires_grad_(True)
                    x_gen.retain_grad()
                    x_guid = x_guid.detach()
                    cond_batch = cond_batch.detach()
                    
                    #Denoising
                    eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                    eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)
                    eps_gen = eps_gen.sample
                    eps_guid = eps_guid.sample

                    pred_x0_guid = (x_guid - sigma * eps_guid) / mu
                    pred_x0_gen = (x_gen - sigma * eps_gen) / mu
                    x_guid = mu_prev * pred_x0_guid + sigma_prev * eps_guid
                    x_gen = mu_prev * pred_x0_gen + sigma_prev * eps_gen

                    x_gen = x_gen.detach()
            #Self Guidance for given number of cycles
            print('Entering the Cycle')
            #opt.use_mutual_self_attention = False
            count = 0
            for idx in range(opt.num_cycles):
                for i, t in enumerate(tqdm(timesteps[opt.start:opt.end])):
                    count += 1
                    i = i + opt.start
                    cond_batch = cond
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                    
                    
                    x_gen.requires_grad_(True)
                    x_gen.retain_grad()
                    x_guid = x_guid.detach()
                    cond_batch = cond_batch.detach()
                    #eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                    
                    if opt.use_mutual_self_attention and i%opt.num_skips == 0:
                        # Case of Mutual Self Attention
                        eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                        # Obtain Keys and Values as needed
                        features = {}
                        for block in self.positions:
                            name = block
                            features[name] = activation[name].detach().cpu()
                        
                        # Temporarily switch neural network layers
                        for layer in self.positions:
                            command = "self.unet." + get_module(layer) + ' = ReplaceKV(replacement_tensor = features[layer])' 
                            exec(command)
                        eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)
                    else:
                        eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                        eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)

                    eps_gen = eps_gen.sample
                    eps_guid = eps_guid.sample

                    E = E_movement(F_guid = F_guid, F_gen = F_gen, opt = opt, timestep = i)
                    if E != 0:
                        E.backward()
                        eps_gen = eps_gen + opt.scale * sigma * x_gen.grad

                    pred_x0_guid = (x_guid - sigma * eps_guid) / mu
                    pred_x0_gen = (x_gen - sigma * eps_gen) / mu
                    x_guid = mu_prev * pred_x0_guid + sigma_prev * eps_guid
                    x_gen = mu_prev * pred_x0_gen + sigma_prev * eps_gen
                    #Save Feats for Tracking
                    if opt.save_feats and count%4 == 0:
                        features_gen[str(count)] = F_gen['2'].detach().to('cpu')
                        features_guid[str(count)] = F_guid['2'].detach().to('cpu')
                    #Replace the layers for the guidance branch
                    if opt.use_mutual_self_attention and i%opt.num_skips == 0:
                        for layer in self.positions:
                            command = "self.unet." + get_module(layer) + ' = unet_layers[layer]' 
                            exec(command)
                    
                    x_gen = x_gen.detach()
                # Invert it back to the start timestep
                if idx < opt.num_cycles - 1:
                    print('Invert ' + str(idx))
                    inverted_x, _ = self.inversion_func(cond, torch.cat((x_gen, x_guid)), opt.save_path, cyclic_inversion_steps={'start':opt.start, 'end':opt.end})
                    
                else:
                    inverted_x, _ = self.inversion_func(cond, torch.cat((x_gen, x_guid)), opt.save_path, cyclic_inversion_steps={'start':0, 'end':opt.end})
                    x_gen = inverted_x[0]
                    x_guid = inverted_x[1]
                    x_gen = torch.unsqueeze(x_gen, 0)
                    x_guid = torch.unsqueeze(x_guid, 0)
                    x_gen.detach()
                    x_gen = x_gen.to(torch.float16) 
                    x_guid = x_guid.to(torch.float16) 
            print('Exiting the Cycle')
            store_processor = CrossAttnStoreProcessor()
            self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = store_processor

            map_size = None

            def get_map_size(module, input, output):
                nonlocal map_size
                map_size = output[0].shape[-2:]
            #Final Set
            with self.unet.mid_block.attentions[0].register_forward_hook(get_map_size):
                for i, t in enumerate(tqdm(timesteps)):
                        #i = i + opt.start
                        cond_batch = cond
                        # Setting up diffusion parameters
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        alpha_prod_t_prev = (
                            self.scheduler.alphas_cumprod[timesteps[i + 1]]
                            if i < len(timesteps) - 1
                            else self.scheduler.final_alpha_cumprod
                        )
                        mu = alpha_prod_t ** 0.5


                        if opt.isStochastic and i >= 350:
                            sigma_t = (((1 - alpha_prod_t_prev)/(1 - alpha_prod_t))**0.5)*(1 - alpha_prod_t/alpha_prod_t_prev)**0.5
                            sigma_prev = (1 - alpha_prod_t_prev - sigma_t**2) ** 0.5
                        else:
                            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                        sigma = (1 - alpha_prod_t) ** 0.5
                        mu_prev = alpha_prod_t_prev ** 0.5

                        x_gen.requires_grad_(True)
                        x_gen.retain_grad()
                        x_guid = x_guid.detach()
                        cond_batch = cond_batch.detach()
                        
                        #Denoising
                        if opt.use_mutual_self_attention:
                            # Case of Mutual Self Attention
                            eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                            # Obtain Keys and Values as needed
                            features = {}
                            for block in self.positions:
                                name = block
                                features[name] = activation[name].detach().cpu()
                            
                            # Temporarily switch neural network layers
                            for layer in self.positions:
                                command = "self.unet." + get_module(layer) + ' = ReplaceKV(replacement_tensor = features[layer])' 
                                exec(command)
                            eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)
                        else:
                            eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                            eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)
                        eps_gen = eps_gen.sample
                        eps_guid = eps_guid.sample


                        if opt.use_sag:
                            # DDIM-like prediction of x0
                            pred_x0 = x_gen/torch.sqrt(alpha_prod_t) - torch.sqrt((1 - alpha_prod_t)/alpha_prod_t)*eps_gen
                            # get the stored attention maps
                            cond_attn = store_processor.attention_probs
                            # self-attention-based degrading of latents
                            degraded_latents = self.sag_masking(
                                pred_x0, cond_attn, map_size, t, eps_gen
                            )
                            # forward and give guidance
                            degraded_pred, _ = self.unet(degraded_latents, t, encoder_hidden_states=cond_batch)
                            eps_gen += opt.sag_scale * (eps_gen - degraded_pred.sample)



                        pred_x0_guid = (x_guid - sigma * eps_guid) / mu
                        pred_x0_gen = (x_gen - sigma * eps_gen) / mu
                        x_guid = mu_prev * pred_x0_guid + sigma_prev * eps_guid
                        x_gen = mu_prev * pred_x0_gen + sigma_prev * eps_gen
                        if opt.isStochastic and i >= 350:
                            x_gen  = x_gen + sigma_t * torch.randn_like(x_gen)
                        if opt.use_mutual_self_attention:
                            for layer in self.positions:
                                command = "self.unet." + get_module(layer) + ' = unet_layers[layer]' 
                                exec(command)
                        x_gen = x_gen.detach()

            x = torch.cat((x_guid, x_gen), dim = 0)
            if save_latents:
                torch.save(x, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return x, features_gen, features_guid
    
    
    
    
    
    def ddim_sample(self, x, cond, opt, save_latents=False, timesteps_to_save=None, activation = None, F_tgt = None):
        save_path = opt.save_dir
        split = torch.split(x, 1)
        x_guid = split[0]
        features_gen = {}
        features_guid = {}
        #Store the original layers in case of mutual attention
        if opt.use_mutual_self_attention:
            print('Entry')
            unet_layers = {}
            for block in self.positions:
                layer = "self.unet." + get_module(block)
                exec("unet_layers[block] = " + layer)
        if opt.use_texture:
            opt.target_texture = opt.target_texture.detach()
        x_gen = split[1]
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i, t in enumerate(tqdm(timesteps)):
                    cond_batch = cond
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                    
                    x_gen.requires_grad_(True)
                    x_gen.retain_grad()
                    x_guid = x_guid.detach()
                    cond_batch = cond_batch.detach()
                    #eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                    if (i>=opt.start and i%opt.num_skips == 0) and opt.use_mutual_self_attention:
                        eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                        # Obtain Keys and Values as needed
                        features = {}
                        for block in self.positions:
                            name = block
                            features[name] = activation[name].detach().cpu()
                        
                        # Temporarily switch neural network layers
                        for layer in self.positions:
                            command = "self.unet." + get_module(layer) + ' = ReplaceKV(replacement_tensor = features[layer])' 
                            exec(command)
                        eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)
                    else:
                        eps_guid, F_guid = self.unet(x_guid, t, encoder_hidden_states=cond_batch)
                        eps_gen, F_gen = self.unet(x_gen, t, encoder_hidden_states=cond_batch)
                    eps_gen = eps_gen.sample
                    eps_guid = eps_guid.sample
                    flag = False
                    # Movement Based Guidance [25, 60)
                    if (i >= opt.start and i < opt.end)  and i%opt.num_skips == 0:
                        flag = True
                        E = E_movement(F_guid = F_guid, F_gen = F_gen, opt = opt)
                        #E = E_movement(F_guid = F_tgt, F_gen = F_gen, opt = opt)
                        E.backward()
                        eps_gen = eps_gen + opt.scale * sigma * x_gen.grad

                    pred_x0_guid = (x_guid - sigma * eps_guid) / mu
                    pred_x0_gen = (x_gen - sigma * eps_gen) / mu
                    x_guid = mu_prev * pred_x0_guid + sigma_prev * eps_guid
                    x_gen = mu_prev * pred_x0_gen + sigma_prev * eps_gen
                    #Save Feats for Tracking
                    if opt.save_feats and i%5 == 0:
                        features_gen[str(t.item())] = F_gen['2'].detach().to('cpu')
                        features_guid[str(t.item())] = F_guid['2'].detach().to('cpu')
                    #Replace the layers for the guidance branch
                    if (i>=opt.start and i%opt.num_skips == 0) and opt.use_mutual_self_attention:
                        for layer in self.positions:
                            command = "self.unet." + get_module(layer) + ' = unet_layers[layer]' 
                            exec(command)
                    
                    x_gen = x_gen.detach()
            x = torch.cat((x_guid, x_gen), dim = 0)
            if save_latents:
                torch.save(x, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return x, features_gen, features_guid

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path, save_path, timesteps_to_save,
                        inversion_prompt='', extract_reverse=False):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path)
        latent = self.encode_imgs(image)

        inverted_x = self.inversion_func(cond, latent, save_path, save_latents=not extract_reverse,
                                         timesteps_to_save=timesteps_to_save)
        latent_reconstruction ,_ = self.ddim_sample(inverted_x, cond, save_path, save_latents=extract_reverse,
                                                 timesteps_to_save=timesteps_to_save)
        rgb_reconstruction = self.decode_latents(latent_reconstruction)

        return rgb_reconstruction  # , latent_reconstruction


class CrossAttnStoreProcessor:
    def __init__(self):
        self.attention_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    


def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img
