import torch
import random
import numpy as np
import torch.nn as nn
from torchvision.transforms import Resize
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
import torchvision.models as models
from einops import rearrange, repeat
from torchvision.utils import make_grid
from kornia.contrib import distance_transform
from ldm.modules.vgg import VGG19_feature_color_torchversion
from ldm.models.diffusion.ddpm import LatentDiffusion, DDPM
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from vton.feature_net import ControlNetModel
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.util import default, log_txt_as_img, ismap, isimage

class TextImageEmbedding(nn.Module):
    def __init__(self, text_embed_dim: int = 768, image_embed_dim: int = 768, out_embed_dim: int = 1536):
        super().__init__()
        self.text_proj = nn.Linear(text_embed_dim, out_embed_dim)
        self.text_norm = nn.LayerNorm(out_embed_dim)
        self.image_proj = nn.Linear(image_embed_dim, out_embed_dim)

    def forward(self, text_embeds: torch.FloatTensor, image_embeds: torch.FloatTensor):
        # text
        out_text_embeds = self.text_proj(text_embeds)
        out_text_embeds = self.text_norm(out_text_embeds)

        # image
        out_image_embeds = self.image_proj(image_embeds)

        return out_image_embeds + out_text_embeds

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

class LatentTryOnDiffusion(LatentDiffusion): # model for MP-VTON
    def __init__(self, first_stage_config, cond_stage_config, cond_text_stage_config, *args, **kwargs):
        super().__init__(first_stage_config, cond_stage_config, *args, **kwargs)
        
        self.cond_text_stage_model = self.instantiate_cond_stage(cond_text_stage_config)
        
        self.text_img_embed = TextImageEmbedding(self.cond_text_stage_model.transformer.config.hidden_size,
                                                 self.cond_stage_model.out_dims,
                                                 self.cond_stage_model.out_dims)
        unet = UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='unet')
        self.feature_net = ControlNetModel.from_unet(unet=unet)
        
        self.vgg = VGG19_feature_color_torchversion(vgg_normal_correct=True)
        self.vgg.load_state_dict(torch.load("pretrain/vgg/vgg19_conv.pth", map_location="cpu"))
        self.vgg.eval()
        
        self.loss_l1_weight = 1e-1
        self.loss_segment_weight = 1e-1
        self.loss_vgg_weight = 1e-3
        
    
    def get_learned_conditioning(self, c, cond_type='image'):
        if cond_type == 'image':
            if self.cond_stage_forward is None:
                if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                    c = self.cond_stage_model.encode(c)
                    if isinstance(c, DiagonalGaussianDistribution):
                        c = c.mode()
                else:
                    c = self.cond_stage_model(c)
            else:
                assert hasattr(self.cond_stage_model, self.cond_stage_forward)
                c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
            return c

        elif cond_type == 'text':
            if self.cond_stage_forward is None:
                if hasattr(self.cond_text_stage_model, 'encode') and callable(self.cond_text_stage_model.encode):
                    c = self.cond_text_stage_model.encode(c)
                    if isinstance(c, DiagonalGaussianDistribution):
                        c = c.mode()
                else:
                    c = self.cond_text_stage_model(c)
            else:
                assert hasattr(self.cond_text_stage_model, self.cond_stage_forward)
                c = getattr(self.cond_text_stage_model, self.cond_stage_forward)(c)
            return c

        
    def get_input(self, batch, return_first_stage_outputs=False, force_c_encode=False, cond_key=None,
                  return_original_cond=False, bs=None, get_mask=False, get_reference=False, get_inpaint=False, get_segment=False, get_annotations=False):

        x = DDPM.get_input(self, batch, 'person')
        cloth_clip = DDPM.get_input(self, batch, 'cloth_clip')
        cloth = DDPM.get_input(self, batch, 'cloth')
        segment = DDPM.get_input(self, batch, 'segment')
        inpaint = DDPM.get_input(self, batch, 'inpaint')
        mask = DDPM.get_input(self, batch, 'mask')
        densepose = DDPM.get_input(self, batch, 'densepose')
        
        caption = batch['caption']
        cloth_caption = batch['cloth_caption']
        annotations = {'caption':caption, 'cloth_caption':cloth_caption}
                
        if bs is not None:
            x = x[:bs]
            cloth_clip = cloth_clip[:bs]
            cloth = cloth[:bs]
            inpaint = inpaint[:bs]
        
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        
        encoder_inpaint = self.encode_first_stage(inpaint)
        z_inpaint = self.get_first_stage_encoding(encoder_inpaint).detach()
        
        encoder_densepose = self.encode_first_stage(densepose)
        z_densepose = self.get_first_stage_encoding(encoder_densepose).detach()
        
        encoder_cloth = self.encode_first_stage(cloth)
        z_cloth = self.get_first_stage_encoding(encoder_cloth).detach()
        
        mask_resize = F.interpolate(mask, size=z.shape[-2:], mode="bilinear", align_corners=False)
        segment_resize = F.interpolate(segment, size=z.shape[-2:], mode="bilinear", align_corners=False)
        
        
        z_addiction = torch.cat((z_inpaint, mask_resize, z_densepose, segment_resize), dim=1)
        z_new = z
        z_diff = z_inpaint
        
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox', "txt"]:
                    xc = batch[cond_key]
                elif cond_key == 'image':
                    xc = cloth_clip
                elif cond_key in ['class_label', 'cls']:
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}
        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        
        out = [z_new, z_diff, z_addiction, c, x]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.append(xrec)
        if return_original_cond:
            out.append(xc)
        if get_inpaint:
            out.append(inpaint)
        if get_mask:
            out.append(mask)
        if get_reference:
            out.append(z_cloth)
        if get_segment:
            out.append(segment)
        if get_annotations:
            out.append(annotations)
        
        return out
        
        
    def shared_step(self, batch, **kwargs):
        # Set segment in z_add (concat with main input set for test)
        # Have to repair code in get_input if want to fusion segment with cloth branch
        
        z_tgt, z_src, z_add, cond, x_tgt,mask, cloth, segment, annotations = self.get_input(batch, get_reference=True, get_annotations=True, get_mask=True, get_segment=True)
        loss, loss_dict = self(z_tgt, z_src, z_add, cond, x_tgt, cloth, mask, segment, annotations, **kwargs)
        return loss, loss_dict
        

    def forward(self, x, x_src, x_add, c, gt, cloth, mask, segment, annotations, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        
        caption = annotations['caption']
        cloth_caption = annotations['cloth_caption']
        
        if self.model.conditioning_key is not None:
            assert c is not None
            
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
                caption = self.get_learned_conditioning(caption, 'text')
                cloth_caption = self.get_learned_conditioning(cloth_caption, 'text')
            
            if self.shorten_cond_schedule:
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
                caption = self.q_sample(x_start=caption, t=tc, noise=torch.randn_like(c.float()))
                cloth_caption = self.q_sample(x_start=cloth_caption, t=tc, noise=torch.randn_like(c.float()))

        loss, loss_dict = self.p_losses(x, c, t, x_add, cloth, caption, cloth_caption)
        
        x_pred = self.sample_output(x, c, t, x_add, cloth, caption, cloth_caption)
        output_pred = self.differentiable_decode_first_stage(x_pred)
        
        prefix = 'train' if self.training else 'val'
        
        l1_loss = self.l1_losses(output_pred, gt)
        loss += l1_loss * self.loss_l1_weight
        loss_dict.update({f'{prefix}/loss_l1': l1_loss})
        
        segment_loss = self.segment_losses(output_pred, gt, mask, segment)
        loss += segment_loss * self.loss_segment_weight
        loss_dict.update({f'{prefix}/loss_segment': segment_loss})
        
        vgg_loss = self.get_vgg_loss(output_pred, gt)
        loss += vgg_loss * self.loss_vgg_weight
        loss_dict.update({f'{prefix}/loss_vgg': vgg_loss})
        
        return loss, loss_dict
            
    def p_losses(self, x_start, img_cond, t, x_add, cloth, caption, cloth_caption, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.first_stage_key == 'inpaint':
            x_noisy = torch.concat((x_noisy, x_add), dim=1)
        
        # Get condition embedding
        feature_samples = self.feature_net(cloth, encoder_hidden_states=cloth_caption, timestep=t)
        cond = self.text_img_embed(img_cond, caption)
        model_output = self.apply_model(x_noisy, t, cond, feature_samples)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def l1_losses(self, pred, gt):
        loss = (pred - gt.detach()).abs()
        return loss.mean()
    
    def segment_losses(self, pred, gt, mask, segment, alpha=10):
        
        def _calcute_weighted_map(uni_map):
            rev_map = 1. - uni_map.float()
            dist_map = distance_transform(rev_map)
            d_max = torch.max(dist_map.flatten(-2, -1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
            
            weighted_map = (dist_map / d_max)
            weighted_map = weighted_map.sum(dim=1, keepdim=True)
            weighted_map[weighted_map > 0] = torch.exp(-alpha * weighted_map[weighted_map > 0])
            return weighted_map
            
        loss = (pred - gt.detach()).abs()
        uni_map = mask.bool() & segment.bool()
        weighted_map = _calcute_weighted_map(uni_map)
        loss = loss * weighted_map
        
        return loss
        
    def get_vgg_loss(self, pred, gt):
        pred_feat = self.vgg(pred, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
        gt_feat = self.vgg(gt, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
        loss_feat = 0
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        for i in range(len(pred_feat)):
            loss_feat += weights[i] * F.l1_loss(pred_feat[i], gt_feat[i].detach())
        return loss_feat
    
    def apply_model(self, x_noisy, t, cond, feature_sample, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, feature_sample=feature_sample)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon
        

    def sample_output(self, x_start, img_cond, t, x_add, cloth, caption, cloth_caption, noise=None):
        default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        z_noisy = torch.cat([x_noisy, x_add], dim=1)
        
        # Get condition embedding
        feature_samples = self.feature_net(cloth, encoder_hidden_states=cloth_caption, timestep=t)
        cond = self.text_img_embed(img_cond, caption)
        
        # Get pred x_start
        model_output = self.apply_model(z_noisy, t, cond, feature_samples) # model_output here must be eps (pred_noise)
        x_denoisy = self.predict_start_from_noise(x_noisy, t, model_output)
        
        return x_denoisy
        

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=50, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=True, **kwargs):
        
        use_ddim = ddim_steps is not None

        log = dict()
        data = self.get_input(batch, 
                              return_first_stage_outputs=True,
                              force_c_encode=True,
                              return_original_cond=True,
                              get_mask=True,
                              get_reference=True,
                              get_inpaint=True,
                              get_segment=True,
                              bs=N
                              )
        z, z_prime, z_add, c, x, xrec, xc, inpaint, mask, cloth, segment  = data
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["mask"] = mask
        log['cloth'] = cloth
        log['inpaint'] = inpaint
        log['segment'] = segment
        # log["reference"]=reference
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                if self.first_stage_key == 'inpaint':
                    ts = torch.full((1,), 999, device=self.device, dtype=torch.long)
                    xt = self.q_sample(z_prime, ts)
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             rest=z, x_T=xt)
                else:
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples


            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                                 ddim_steps=ddim_steps, x0=z[:N, :4], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                mask = 1. - mask
                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                                 ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
