import os

from PIL import Image
import torch
import numpy as np
import torch.utils.data as data 
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import CLIPProcessor
from typing import Literal, Tuple,List
import json


import random
import os.path as osp


class VTHDDataset(data.Dataset):
    """
        Dataset for VTON HD
    """
    
    def __init__(self, 
                 dataroot:str, 
                 image_size:Tuple[int, int] = (512, 768), 
                 cloth_size:Tuple[int, int] = (512, 384),
                 max_person:int = 5, 
                 mode:Literal["train", "test"] = "train", 
                 order:Literal["paired", "unpaired"] = "paired"):
        self.root = dataroot
        self.mode = mode
        self.datalist = mode + '_pairs.txt'
        self.fine_height = image_size[0]
        self.fine_width = image_size[1]
        self.max_person = max_person
        self.cloth_size = cloth_size
        self.data_path = osp.join(dataroot, mode)
        self.order = order
        
        # transforms
        self.crop_size = (self.fine_height, self.fine_width)
        self.toTensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.clip_transform = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.transform_mask = transforms.Compose([
            transforms.ToTensor()
        ])
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        
        # annotation
        with open(
            os.path.join(dataroot, mode, "vitonhd_" + mode + "_tagged.json"), "r"
        ) as file1:
            data = json.load(file1)
        
        annotation_list = [
            "sleeveLength",
            "neckLine",
            "details",
            "item",
        ]
        
        self.annotation_pair = {}
        self.nums_annotation = {}
        for k, v in data.items():
            for elem in v:
                annotation_str = ""
                for template in annotation_list:
                    for tag in elem["tag_info"]:
                        if (
                            tag["tag_name"] == template
                            and tag["tag_category"] is not None
                        ):
                            annotation_str += tag["tag_category"]
                            annotation_str += " "
                self.annotation_pair[elem["file_name"]] = annotation_str
                #TODO: NEED TO SET number of person
                # self.nums_annotation[elem["file_name"]] = elem['per_info']['nums']
        # get list of filename
        im_names = []
        c_names = []
        
        with open(osp.join(dataroot, self.datalist), "r") as f:
            for line in f.readlines():
                if mode == 'train':
                    im_name, c_name = line.strip().split()
                else:
                    if order == 'paired':
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()
                
                im_names.append(im_name)
                c_names.append(c_name)
        
        self.im_names = im_names
        self.c_names = c_names
        
        
    
    def __len__(self):
        return len(os.listdir(osp.join(self.data_path, 'image')))
    
    def __getitem__(self, index):
        filename = self.im_names[index]
        filename_cloth = self.c_names[index]
        
        # annotation
        if filename_cloth in self.annotation_pair:
            cloth_annotation = self.annotation_pair[filename_cloth]
        else:
            cloth_annotation = "shirts"
            
        nums_person = 2
        
        # person image
        person = Image.open(osp.join(self.data_path, 'image', filename))
        person = transforms.Resize(self.crop_size, interpolation=2)(person)
        person = self.transform(person)
        
        # inpainting image
        inpaint = Image.open(osp.join(self.data_path, 'agnostic', filename))
        inpaint = transforms.Resize(self.crop_size, interpolation=2)(inpaint)
        inpaint = self.transform(inpaint)
        
        # mask inpainting
        mask = Image.open(osp.join(self.data_path, 'agnostic-mask', os.path.splitext(filename)[0] + '.png')).convert('L')
        mask = transforms.Resize(self.crop_size, interpolation=0)(mask)
        mask = self.transform_mask(mask).long()
        
        # cloth image
        cloth = Image.open(osp.join(self.data_path, 'cloth', filename_cloth))
        cloth = transforms.Resize(self.cloth_size, interpolation=2)(cloth)
        cloth_trim = self.clip_transform(images=cloth, return_tensors="pt").pixel_values 
        cloth = self.transform(cloth)
        
        
        densepose_map = Image.open(osp.join(self.data_path, 'densepose', filename))
        densepose_map = transforms.Resize(self.crop_size, interpolation=0)(densepose_map)
        densepose_map = self.transform(densepose_map)
        
        
        segment = Image.open(osp.join(self.data_path, 'segment', os.path.splitext(filename)[0] + '.png')).convert('L')
        segment = transforms.Resize(self.crop_size, interpolation=0)(segment)
        segment = torch.from_numpy(np.array(segment)).long()
        segment = F.one_hot(segment, 256).permute(2, 0, 1).to(float)
        segment = segment[1:]
        select = torch.sum(segment, dim = (1, 2)) > 0
        segment = segment[select] 

        if (segment.shape[0] < self.max_person):
            padding = torch.zeros((self.max_person-segment.shape[0], *segment.shape[1:]))
            segment = torch.cat((segment, padding), dim=0)
        
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
                densepose_map = TF.affine(
                    densepose_map, angle=0, translate=[0, 0], scale=scale_val, shear=0
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
                densepose_map = TF.affine(
                    densepose_map,
                    angle=0,
                    translate=[
                        shift_valx * densepose_map.shape[-1],
                        shift_valy * densepose_map.shape[-2],
                    ],
                    scale=1,
                    shear=0,
                )
                
        
        
        result = {}
        result['person'] = person
        result['cloth_clip'] = cloth_trim
        result['cloth'] = cloth
        result['segment'] = segment
        result['inpaint'] = inpaint
        result['mask'] = mask
        result['densepose'] = densepose_map
        result["caption"] = f"There are {nums_person} people wearing " + cloth_annotation
        result["caption_cloth"] = "a photo of " + cloth_annotation
        result["annotation"] = cloth_annotation
        
        return result
