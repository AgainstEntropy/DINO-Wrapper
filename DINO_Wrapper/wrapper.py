from typing import Optional, Callable, Union
from omegaconf import OmegaConf
from copy import deepcopy

import torch
from torch import nn

from .configs import DinoConf


class DinoWrapper(nn.Module):
    def __init__(self, 
                 model: nn.Module,
                #  configs: Optional[Union[OmegaConf, dict]] = None,
                 configs: Optional[DinoConf] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


        self.cfgs = configs if configs is not None else DinoConf()
        self.preprocess_func = configs.preprocess_func

        self.teacher, self.student = None, None
        self.prepare_model(model)

        self.augmentation = None
        self.prepare_aug(configs)

        self.dino_loss = None
        self.prepare_loss(configs)


        raise NotImplementedError
    
    def forward(self, input, mode:str = 'train'):
        if mode == 'train':
            return self.train_forward(input)
        elif mode == 'val':
            return self.val_forward(input)
        else:
            raise NotImplementedError
    
    def train_forward(self, input):
        teacher_images, student_images = self.augmentation(input)

        teacher_crop_num = teacher_images.shape[1]
        student_crop_num = student_images.shape[1]

        teacher_features = []  # 2 * (C, H, W)
        student_features = []  # 8 * (C, H, W)
        for i in range(teacher_crop_num):
            features = self.teacher(teacher_images[:, i])
            teacher_features.append(features[0])
        teacher_features = torch.stack(teacher_features, dim=0)  # (2, BS, C, H, W)
        
        for i in range(student_crop_num):
            features, pos = self.student(student_images[:, i])
            student_features.append(features[0])
        student_features = torch.stack(student_features, dim=0)  # (8, BS, C, H, W)
        
        raise NotImplementedError

    
    def infer_forward(self, input):
        return self.teacher(input)
    
        raise NotImplementedError


    def prepare_model(self, model: nn.Module):
        self.student = model
        self.teacher = deepcopy(model)
        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    def prepare_aug(self, cfgs: OmegaConf):
        raise NotImplementedError
    
    def prepare_loss(self, model: nn.Module):
        raise NotImplementedError

    def preprocess(self, input):
        return self.preprocess_func(input)
    