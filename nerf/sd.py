from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import repeat
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd 


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad) 
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype) # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.1', hf_key=None, opt=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.opt = opt

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
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", cache_dir = opt.cache_dir).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer", cache_dir = opt.cache_dir)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", cache_dir = opt.cache_dir).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", cache_dir = opt.cache_dir).to(self.device)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", cache_dir = opt.cache_dir)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * opt.t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, q_unet = None, pose = None, shading = None, grad_clip = None, as_latent = False, t5 = False):
        
        # interp to 512x512 to be fed into vae.
        assert torch.isnan(pred_rgb).sum() == 0, print(pred_rgb)
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)
        elif self.opt.latent == True:
            latents = pred_rgb
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)        

        if t5: # Anneal time schedule
            t = torch.randint(self.min_step, 700 + 1, [1], dtype=torch.long, device=self.device)
        else:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        # TODO: noise calculating part
        
        if self.opt.grad_method == 'estimate':
            w = (1 - self.alphas[t])
            sqrt_alpha_prod = self.alphas[t] ** 0.5
            sigmat = (1 - self.alphas[t]) ** 0.5

            num_particles = self.opt.num_estimate_samples 
            dimprod = torch.prod(torch.tensor(latents.shape[1:]))
            kernel_sig = torch.sqrt(dimprod)*sigmat*self.opt.kernel_sig_scale
            # kernel_sig = torch.sqrt(dimprod)*torch.sqrt(sigmat)*self.opt.kernel_sig_scale


            with torch.no_grad():
                num_noise = num_particles if self.opt.x_star_included else num_particles + 1
                noise = torch.randn(size = [num_noise, *latents.shape[1:]], device=latents.device)

                latents_noisy = self.scheduler.add_noise(latents, noise, t)

                latent_model_input = torch.cat([latents_noisy[:num_particles]] * 2)
                # turn text_embeddings (2,h,w) into (2*num_particles, h, w)
                text_embeddings = repeat(text_embeddings, 'b h w -> (b p) h w', p=num_particles)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                x_star = latents_noisy[-1].unsqueeze(0)
                difference = (latents_noisy[:num_particles] - x_star)
                kernel_value = torch.exp(-torch.sum(difference**2, dim=(1,2,3))/(2*kernel_sig**2))[..., None, None, None]
                # print(kernel_value)

            grad = torch.sum(kernel_value * (noise_pred/sigmat + difference/kernel_sig**2), dim=0)/torch.sum(kernel_value, dim=0)*sigmat*w/sqrt_alpha_prod

            weight = (sigmat, sqrt_alpha_prod, w)
                # weight = 0.
            loss = SpecifyGradient.apply(latents, grad.unsqueeze(0))
        
        elif self.opt.grad_method == 'sde':
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance (high scale from paper!)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                # An error in original implimentation
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)



                # if self.opt.sds is False:
                #     if q_unet is not None:
                #         if pose is not None:
                #             noise_pred_q = q_unet(latents_noisy, t, c = pose, shading = shading).sample
                #         else:
                #             raise NotImplementedError()

                #         # TODO: what is this v_pred? 
                #         if self.opt.v_pred:
                #             sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5
                #             sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                #             while len(sqrt_alpha_prod.shape) < len(latents_noisy.shape):
                #                 sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                #             sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(self.device)[t]) ** 0.5
                #             sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                #             while len(sqrt_one_minus_alpha_prod.shape) < len(latents_noisy.shape):
                #                 sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                #             noise_pred_q = sqrt_alpha_prod * noise_pred_q + sqrt_one_minus_alpha_prod * latents_noisy


            # w(t), sigma_t^2
            w = (1 - self.alphas[t])
            
            # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
            # TODO: change the loss computation
            if self.opt.sds:
                grad = w * (noise_pred - noise)
            else:
                # grad = w * (noise_pred - noise_pred_q)
                sqrt_alpha_prod = self.alphas[t] ** 0.5
                sigmat = (1 - self.alphas[t]) ** 0.5
                # w = sigmat/sqrt_alpha_prod
                # snr = sqrt_alpha_prod/sigmat
                # eta_1 = self.opt.eta_1/2 if not t5 else self.opt.eta_1/10
                # eta_1 = snr
                # grad = eta_1 * noise_pred - snr*noise + torch.sqrt(2*snr*eta_1)*torch.randn_like(noise, device=noise.device) * 0.01

                # # dreamfusion setting, wt = alpha_t*sigma_t/eta1
                
                # grad = sqrt_alpha_prod*sigmat*(noise_pred - noise)

                # flow-to-theta implementation
                # 
                # w = w*sqrt_alpha_prod
                grad =  w * (noise_pred ) / sqrt_alpha_prod
                # 1/snr + 1 weighting
                # grad =  (noise_pred) *  (w /sqrt_alpha_prod + 1)
                # weight = torch.sqrt(sigmat / sqrt_alpha_prod  * w * 2)*0.01



                weight = (sigmat, sqrt_alpha_prod, w/sqrt_alpha_prod)
                # weight = 0.
                loss = SpecifyGradient.apply(latents, grad)
        else:
            raise NotImplementedError()


        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        if grad_clip is not None:
            grad = grad.clamp(-grad_clip, grad_clip)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        #TODO: ?
        

        pseudo_loss = torch.mul((w*noise_pred).detach(), latents.detach()).detach().sum()

        return loss, pseudo_loss, latents, weight


    def train_multi_step(self, text_embeddings, pred_rgb, guidance_scale=100, q_unet = None, pose = None, shading = None, grad_clip = None, as_latent = False, t5 = False, latents_noisy = None, t=None):
        
        # interp to 512x512 to be fed into vae.
        assert torch.isnan(pred_rgb).sum() == 0, print(pred_rgb)
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)
        elif self.opt.latent == True:
            latents = pred_rgb
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)        

        if t is None:
            if t5: # Anneal time schedule
                t = torch.randint(self.min_step, 500 + 1, [1], dtype=torch.long, device=self.device)
            else:
                # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
                t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        # multi-step
        w = (1 - self.alphas[t])
        sqrt_alpha_prod = self.alphas[t] ** 0.5
        sigmat = (1 - self.alphas[t]) ** 0.5

        # predict the noise residual with unet, NO grad!
        # TODO: noise calculating part
        if  latents_noisy is None:
            


            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                for i in range(self.opt.multi_step_m):

                    latent_model_input = torch.cat([latents_noisy] * 2)
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    # perform guidance (high scale from paper!)
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    # 
                    # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents_noisy = latents_noisy - self.opt.eta_1 * w/sqrt_alpha_prod * noise_pred + self.opt.noise_scaler_xt*torch.sqrt(2*sigmat*w/sqrt_alpha_prod * self.opt.eta_1)*torch.randn_like(noise_pred, device=noise.device)

            # grad = sqrt_alpha_prod/sigmat**2 * (sqrt_alpha_prod * latents - latents_noisy)
            # loss = torch.mean((sqrt_alpha_prod*latents - latents_noisy)**2/sigmat**2 * 0.5)


            # clip grad for stable training?
            # grad = grad.clamp(-10, 10)
            # if grad_clip is not None:
            #     grad = grad.clamp(-grad_clip, grad_clip)
            # grad = torch.nan_to_num(grad)

            # since we omitted an item in grad, we need to use the custom function to specify the gradient
            #TODO: ?
            

            pseudo_loss = torch.mul((w*noise_pred).detach(), latents.detach()).detach().sum()

        else:
            pseudo_loss = 0

        # loss = torch.sum((sqrt_alpha_prod*latents - latents_noisy)**2/sigmat**2 * 0.5)
        grad = sqrt_alpha_prod/sigmat**2 * (sqrt_alpha_prod * latents - latents_noisy)
        loss = SpecifyGradient.apply(latents, grad)

        return latents_noisy, pseudo_loss, latents, (None,None,None), loss, t



    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
