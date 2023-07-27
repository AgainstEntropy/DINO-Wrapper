# DINO-Wrapper

A wrapper for any backbone to integrate DINO, a self-supervised method, with few lines of code.

This PyTorch implementation heavily references the repositories of [DINO](https://github.com/facebookresearch/dino/) and [DINO v2](https://github.com/facebookresearch/dinov2/).

If you find this repository useful, please consider giving a star :star: and citation :t-rex:

DINO: 
[[`Repo`](https://github.com/facebookresearch/dino/)] 
[[`Blog`](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training)] 
[[`Paper`](https://arxiv.org/abs/2104.14294)] 

DINO v2: 
[[`Repo`](https://github.com/facebookresearch/dinov2/)] 
[[`Blog`](https://ai.facebook.com/blog/dino-v2-computer-vision-self-supervised-learning/)] 
[[`Paper`](https://arxiv.org/abs/2304.07193)] 

## Coming Updates

- [ ] Use intermediate feature for computing dino loss

- [ ] Cosider use EMA implementation from [timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py), [MS](https://github.com/microsoft/unilm/blob/master/edgelm/fairseq/models/ema/ema.py), or [ema-pytorch](https://github.com/lucidrains/ema-pytorch).

## Usage

Without DINO support:

```python
model, images = ...
original_output = model(images)
loss = criterion(original_output)
loss.backward()
...
```

With DINO support:

```python
from DINO_Wrapper import DINOWrapper

model, images = ...
dino_model = DINOWrapper(model)  # DINOv1 by default
original_output, dino_loss = dino_model(images)
loss = criterion(original_output) + dino_loss
loss.backward()
...
```

Note that we expect the `original_output` is in the shape of `[B, D]`, where `B` is the batch size and `D` is the dimension of feature vectors, so that we can directly compute the DINO loss on the `original_output` by default.

## Custome Configurations Supported

```python
from DINO_Wrapper import DINOWrapper, DINOConf

dino_cfgs = DINOConf(
    version='v2',
    local_crops_number = 2,
    global_crops_scale = (0.9, 1.), 
    global_crop_size = (480, 640),
    local_crops_number = 8,
    local_crops_scale = (0.6, 0.7), 
    local_crop_size = (240, 320),
    preprocess_func = ...,
    len_train_loader = len(train_loader),
)

model, images = ...
dino_model = DINOWrapper(model, dino_cfgs)  # using DINOv2 now
original_output, dino_loss = dino_model(images)
loss = criterion(original_output) + dino_loss
loss.backward()
...
```

## Citations

```
@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}

@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```