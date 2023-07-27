from typing import Optional, Sequence
from copy import deepcopy

import torch
from torch import nn, Tensor
from torchvision.models.feature_extraction import create_feature_extractor

from .configs import DINOConf
from .loss import DINOLossV1
from .data import DINOAugV1
from .utils import cosine_scheduler, MaybeToPIL


class DINOWrapper(nn.Module):
    def __init__(self, 
                 model: nn.Module,
                 configs: Optional[DINOConf] = None,
                 *args, **kwargs) -> None:
        super().__init__()


        self.cfgs = configs if configs is not None else DINOConf()
        self.preprocess_func = configs.preprocess_func

        self.teacher, self.student = None, None
        self.prepare_model(model)
        self.prepare_ema()

        self.augmentation = None
        self.prepare_aug()

        self.dino_loss = None
        self.prepare_loss()
    
    def forward(self, input, epoch: int = 0, mode: str = 'train'):
        if mode == 'train':
            return self.train_forward(input, epoch)
        elif mode == 'val':
            return self.infer_forward(input)
        else:
            raise NotImplementedError
    
    def train_forward(self, input: Tensor, epoch: int):
        input_pil = [self.preprocess_data(pic) for pic in input]
        aug_dict_list: list[dict[str, list]] = [self.augmentation(pil) for pil in input_pil]

        forward_feature, dino_features = self.dino_forward(aug_dict_list)

        return forward_feature, self.dino_loss(*dino_features, epoch)
    
    def dino_forward(self, aug_dict_list: Sequence[dict]):
        batch_size = len(aug_dict_list)
        global_num = len(aug_dict_list[0]['global_crops']) # 2 * (C, H, W)
        local_num = len(aug_dict_list[0]['local_crops']) # (2+8) * (C, H, W)

        global_crops = [torch.stack([aug_dict_list[b]['global_crops'][n] \
                                     for b in range(batch_size)]) \
                        for n in range(global_num)]
        local_crops = [torch.stack([aug_dict_list[b]['local_crops'][n] \
                                     for b in range(batch_size)]) \
                        for n in range(local_num)]

        teacher_features = []  # 2 * (BS, C, H, W)
        student_features = []  # (2+8) * (BS, C, H, W)
        # only global crops pass through the teacher
        for g_crop in global_crops:
            t_features = self.teacher(g_crop)
            t_features = self.preprocess_feature(t_features)
            teacher_features.append(t_features)
        teacher_features = torch.stack(teacher_features, dim=0)  # (2, BS, C [, H, W])
        
        # all crops pass through the student 
        for crop in global_crops + local_crops:
            s_features = self.student(crop)
            s_features = self.preprocess_feature(s_features)
            student_features.append(s_features)
        student_features = torch.stack(student_features, dim=0)  # (2+8, BS, C [, H, W])
        
        forward_feature = student_features[:global_num].mean(dim=0)

        return forward_feature, (student_features, teacher_features)

    def infer_forward(self, input):
        with torch.inference_mode():
            x = self.augmentation.normalize(input)
            t_features = self.teacher(x)
            return self.preprocess_feature(t_features)
    
    def ema_update(self, it:int):
        # EMA update for the teacher
        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def prepare_model(self, model: nn.Module):
        setattr(self.cfgs, 'out_dim', model.fc.in_features)
        model.fc = nn.Identity()

        if self.cfgs.return_layer is None:
            self.student = model
        else:
            self.student = create_feature_extractor(
                model, 
                return_nodes=[self.cfgs.return_layer]
            )
        self.teacher = deepcopy(self.student)
        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    def prepare_ema(self):
        if self.cfgs.version == 'v1':
            # momentum parameter is increased to 1. during training with a cosine schedule
            self.momentum_schedule = cosine_scheduler(self.cfgs.momentum_teacher, 1,
                                                      self.cfgs.epochs, self.cfgs.len_train_loader)
        elif self.cfgs.version == 'v2':
            raise NotImplementedError
        else:
            raise NotImplementedError

    
    def prepare_aug(self):
        if self.cfgs.version == 'v1':
            self.augmentation = DINOAugV1(
                self.cfgs.global_crops_scale, 
                self.cfgs.local_crops_scale, 
                self.cfgs.local_crops_number
            )
        elif self.cfgs.version == 'v2':
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.preprocess_data = MaybeToPIL()

        
    def prepare_loss(self):
        if self.cfgs.version == 'v1':
            self.dino_loss = DINOLossV1(
                self.cfgs.out_dim,
                self.cfgs.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
                self.cfgs.warmup_teacher_temp,
                self.cfgs.teacher_temp,
                self.cfgs.warmup_teacher_temp_epochs,
                self.cfgs.epochs,
            )
        elif self.cfgs.version == 'v2':
            raise NotImplementedError
        else:
            raise NotImplementedError


    def preprocess_feature(self, input):
        if self.cfgs.return_layer is not None:
            input = input[self.cfgs.return_layer]
        return self.preprocess_func(input)
    