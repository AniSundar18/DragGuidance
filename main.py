import os
from PIL import Image
from preprocess import Preprocess
from tqdm import tqdm, trange
import torchvision
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as T
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from torch.linalg import norm
from torchvision.utils import make_grid



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

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def set_hooks(model, positions, pipe):
    for block in positions:
        name = block
        ext = get_module(block) + f".register_forward_hook(get_activation('" + name + "'))"
        print("pipe.unet." + ext)
        exec("pipe.unet." + ext)

def get_inversion(opt, pipe, cond):
    image = pipe.load_img(opt.data_path)
    print('Img: ' , image.shape)
    latent = pipe.encode_imgs(image)
    inverted_x, F_tgt = pipe.inversion_func(cond, latent, opt.save_path)
    return inverted_x, F_tgt

def get_condition(opt, pipe):
    #Get representation for guiding text/image
    if opt.condition_type == 'text':
        cond = pipe.get_text_embeds(opt.inversion_prompt, "")[1].unsqueeze(0)
    elif opt.condition_type == 'image':
        image = pipe.load_img(opt.data_path)[0]
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
        tensor_image = transform(image).to('cuda').unsqueeze(0)
        print(tensor_image.shape)
        opt.guidance_image = tensor_image
        if opt.use_texture:
            opt.target_texture = pipe.dino(tensor_image).pooler_output
        cond = pipe.sd_pipe._encode_image(tensor_image, device = 'cuda', num_images_per_prompt = 1, do_classifier_free_guidance=False).to('cuda')
    return cond

def get_model(opt):
    pipe = Preprocess(opt)
    #Initialize and return class containing UNet and Autoencoders
    return pipe

if __name__ == "__main__":
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    #Data Related Arguments
    parser.add_argument('--data_path', type=str,
                        default='data/horse.jpg')
    parser.add_argument('--save_dir', type=str, default='latents')
    
    #Stable Diffusion Model
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '1.4', '2.1', 'depth', 'variations', 'sag'],
                        help="stable diffusion version")
    
    #Experimental Details
    parser.add_argument('--isStochastic', type=bool, default=False, help="DDPM for last how many ever steps for improved quality")
    parser.add_argument('--save_feats', type=bool, default=False, help="Save Features for tracking")
    parser.add_argument('--use_mutual_self_attention', type=bool, default=True, help="Use mutual self attention")
    parser.add_argument('--guidance', type=float, default=None, help="Classifier free guidance value, None is no guidance")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_texture', type=bool, default=False)
    parser.add_argument('--steps', type=int, default=499)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--condition_type', type=str, default='image', choices=['text', 'image'])
    parser.add_argument('--inversion_prompt', type=str, default='', help="In case of text conditioned diffusion")
    parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
    parser.add_argument('--scale', type=float, default=160, help="Scale of Gradients")

    #Cyclic Dragging
    parser.add_argument('--isCyclic', type=bool, default=True, help="Perform cyclic form of updates")
    parser.add_argument('--start', type=int, default=100)
    parser.add_argument('--end', type=int, default=200)#220
    parser.add_argument('--num_cycles', type=int, default=1)
    parser.add_argument('--num_skips', type=int, default=1)

    #Self-Attention Guidance
    parser.add_argument('--use_sag', type=bool, default=True, help="Use Self-Attention Guidance")
    parser.add_argument('--sag_scale', type=int, default=1)

    opt = parser.parse_args()
    

    
    opt.handle_points = [(26*8, 12*8)]
    opt.target_points = [(26*8, 20*8)]

   

    
    extraction_path_prefix = "_reverse" if opt.extract_reverse else "_forward"
    save_path = os.path.join(opt.save_dir + extraction_path_prefix, os.path.splitext(os.path.basename(opt.data_path))[0])
    os.makedirs(save_path, exist_ok=True)
    opt.save_path = save_path

    # Setup the pipeline, condition and model
    pipe = get_model(opt)
    if opt.use_mutual_self_attention:
        set_hooks(model=pipe, positions = pipe.positions, pipe = pipe)

    cond = get_condition(opt = opt, pipe = pipe)
    inverted_x, F_tgt = get_inversion(opt = opt, pipe = pipe, cond = cond)


    # Make everything into batches of 2, 0: Guidance, 1: Generation
    inverted_x = torch.cat((inverted_x, inverted_x), dim = 0)

    # Perform editing
    if opt.isCyclic:
        if opt.use_sag:
            latent_reconstruction, features_gen, features_guid = pipe.ddim_cyclic_sag_sample(x = inverted_x, cond = cond, opt = opt, activation = activation, F_tgt = F_tgt)
        else:
            latent_reconstruction, features_gen, features_guid = pipe.ddim_cyclic_sample(x = inverted_x, cond = cond, opt = opt, activation = activation, F_tgt = F_tgt)
    else:
        latent_reconstruction, features_gen, features_guid = pipe.ddim_sample(x = inverted_x, cond = cond, opt = opt, activation = activation, F_tgt = F_tgt)

    #Finishing touches
    with torch.no_grad():
        rgb_reconstruction = pipe.decode_latents(latent_reconstruction)

    n = opt.data_path.split('/')[1].split('.')[0]
    T.ToPILImage()(rgb_reconstruction[0]).save(os.path.join(save_path, n + f'_orig.jpg'))
    T.ToPILImage()(rgb_reconstruction[1]).save(os.path.join(save_path, n + f'_drag.jpg'))
    if opt.save_feats:
        torch.save(features_guid, os.path.join(save_path, n+ f'_orig_recon_dict.pth'))
        torch.save(features_gen, os.path.join(save_path, n+ f'_drag_recon_dict.pth'))

