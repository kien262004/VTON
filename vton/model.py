import torch
import random
import torch.nn as nn
from torchvision.transforms import Resize
from diffusers import UNet2DConditionModel
from ldm.models.diffusion.ddpm import LatentDiffusion, DDPM
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from vton.feature_net import ControlNetModel
from ldm.util import default

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
        # TODO: Need implement feature_net
        
        self.cond_text_stage_model = self.instantiate_cond_stage(cond_text_stage_config)
        
        self.text_img_embed = TextImageEmbedding(self.cond_text_stage_model.transformer.config.hidden_size,
                                                 self.cond_stage_model.out_dim,
                                                 self.cond_stage_model.out_dim)
        unet = UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-1')
        self.feature_net = ControlNetModel.from_unet(unet=unet)
        
    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)
    
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
            segment = segment[:bs]
            inpaint = inpaint[:bs]
            mask = mask[:bs]
        
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        encoder_inpaint = self.encode_first_stage(inpaint)
        z_inpaint = self.get_first_stage_encoding(encoder_inpaint).detach()
        encoder_densepose = self.encode_first_stage(densepose)
        z_densepose = self.get_first_stage_encoding(encoder_densepose).detach()
        mask_resize = Resize((z.shape[-1], z.shape[-1]))(mask)
        segment = Resize((z.shape[-1], z.shape[-1]))(segment)
        
        z_addiction = torch.cat((z_inpaint, mask_resize, z_densepose, segment), dim=1)
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
            out.append(z_inpaint)
        if get_mask:
            out.append(mask)
        if get_reference:
            out.append(cloth)
        if get_segment:
            out.append(segment)
        if get_annotations:
            out.append(annotations)
        
        return out
        
        
    def shared_step(self, batch, **kwargs):
        # Set segment in z_add (concat with main input set for test)
        # Have to repair code in get_input if want to fusion segment with cloth branch
        
        z_tgt, z_src, z_add, cond, x_tgt, cloth, annotations = self.get_input(batch, get_reference=True, get_annotations=True)
        loss, loss_dict = self(z_tgt, z_src, z_add, cond, x_tgt,  cloth, annotations, **kwargs)
        return loss, loss_dict
        

    def forward(self, x, x_src, x_add, c, gt, cloth, annotations, *args, **kwargs):
        self.opt.params = self.params
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


        # if self.u_cond_prop < self.u_cond_percent:
        #     self.opt.params = self.params_with_white
        #     loss, loss_dict = self.p_losses(x, c, t, x_add, cloth, caption, cloth_caption)

        # else:
        loss, loss_dict = self.p_losses(x, c, t, x_add, cloth, caption, cloth_caption)
        
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

        logvar_t = self.logvar[t].to(self.device)
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

    def apply_model(self, x_noisy, t, cond, feature_sample, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, cond, feature_sample)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon
        

    def sample_hijack(self, x_start, cond, t, controlnet_cond_f=None, controlnet_cond_b=None, skeleton_cf=None,
                      skeleton_cb=None, skeleton_p=None, noise=None):
        pass

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=True, **kwargs):
        pass
