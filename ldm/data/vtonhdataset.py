import os

from PIL import Image
import torch
import torch.utils.data as data 
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import CLIPImageProcessor

import random
import os.path as osp


class VTHDDataset(data.Dataset):
    """
        Dataset for VTON HD
    """
    
    def __init__(self, dataroot, image_size=512, max_person=5, mode='train'):
        self.root = dataroot
        self.mode = mode
        self.datalist = mode + '_pairs.txt'
        self.fine_height = image_size
        self.fine_width = int(image_size / 256 * 256)
        self.max_person = max_person
        self.data_path = osp.join(dataroot, mode)
        # transforms
        self.crop_size = (self.fine_height, self.fine_width)
        self.toTensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.clip_transform = CLIPImageProcessor()
        self.transform_mask = transforms.Compose([
            transforms.ToTensor()
        ])
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        
    
    def __len__(self):
        len(os.listdir(osp(self.data_path, 'image')))
    
    def __getitem__(self, index):
        filename = f'{index:05d}_00.jpg'
        # person image
        person = Image.open(osp(self.data_path, 'image', filename))
        person = transforms.Resize(self.crop_size, interpolation=2)(person)
        person = self.transform(person)
        
        # inpainting image
        inpaint = Image.open(osp(self.data_path, 'agnostic', filename))
        inpaint = transforms.Resize(self.crop_size, interpolation=2)(inpaint)
        inpaint = self.transform(inpaint)
        
        # mask inpainting
        mask = (person - inpaint)[:1]
        mask[mask != 0] = 1 
        
        # cloth image
        cloth = Image.open(osp(self.data_path, 'cloth', filename))
        
        
        # dense pose image (NEED MODIFINE)
        # densepose_map = Image.open(osp(self.data_path, 'image-densepose', filename))
        # densepose_map = self.transform_mask(densepose_map)
        
        segment = Image.open(osp(self.data_path, 'segment', filename)).convert('L')
        segment = transforms.Resize(self.crop_size, interpolation=2)(segment)
        segment = self.transform_mask(segment)
        segment = F.one_hot(segment, segment.shape[0]).dtype(float)[1:]
        if (segment.shape[0] < self.max_person):
            segment = torch.cat((segment, torch.zeros((self.max_person-segment.shape[0], *segment.shape[1:]))), dim=0)
        
        if self.mode == 'train':
            if random.random() > 0.5:
                person = self.flip_transform(person)
                segment = self.flip_transform(segment)
                cloth = self.flip_transform(cloth)
                mask = self.flip_transform(mask)
            
            if random.random()>0.5:
                color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5)
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,color_jitter.hue)
                
                person = TF.adjust_contrast(person, c)
                person = TF.adjust_brightness(person, b)
                person = TF.adjust_hue(person, h)
                person = TF.adjust_saturation(person, s)
                
                inpaint = TF.adjust_contrast(inpaint, c)
                inpaint = TF.adjust_brightness(inpaint, b)
                inpaint = TF.adjust_hue(inpaint, h)
                inpaint = TF.adjust_saturation(inpaint, s)

                cloth = TF.adjust_contrast(cloth, c)
                cloth = TF.adjust_brightness(cloth, b)
                cloth = TF.adjust_hue(cloth, h)
                cloth = TF.adjust_saturation(cloth, s)
        
            if random.random() > 0.5:
                scale_val = random.uniform(0.8, 1.2)
                person = TF.affine(
                    person, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                mask = TF.affine(
                    mask, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                segment = TF.affine(
                    segment, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                inpaint = TF.affine(
                    inpaint, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )


            if random.random() > 0.5:
                shift_valx = random.uniform(-0.2, 0.2)
                shift_valy = random.uniform(-0.2, 0.2)
                person = TF.affine(
                    person,
                    angle=0,
                    translate=[shift_valx * person.shape[-1], shift_valy * person.shape[-2]],
                    scale=1,
                    shear=0,
                )
                mask = TF.affine(
                    mask,
                    angle=0,
                    translate=[shift_valx * mask.shape[-1], shift_valy * mask.shape[-2]],
                    scale=1,
                    shear=0,
                )
                segment = TF.affine(
                    segment,
                    angle=0,
                    translate=[
                        shift_valx * segment.shape[-1],
                        shift_valy * segment.shape[-2],
                    ],
                    scale=1,
                    shear=0,
                )
                inpaint = TF.affine(
                    inpaint,
                    angle=0,
                    translate=[
                        shift_valx * inpaint.shape[-1],
                        shift_valy * inpaint.shape[-2],
                    ],
                    scale=1,
                    shear=0,
                )
        
        cloth_trim =  self.clip_transform(images=cloth, return_tensors="pt").pixel_values
        
        result = {}
        result['person'] = person
        result['cloth_clip'] = cloth_trim
        result['cloth'] = cloth
        result['segment'] = segment
        result['inpaint'] = inpaint
        result['mask'] = mask
        
        return result
