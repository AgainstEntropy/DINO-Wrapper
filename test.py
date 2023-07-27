import torch
import torchvision

from DINO_Wrapper import DINOWrapper, DINOConf


if __name__ == '__main__':
    dino_cfgs = DINOConf(
        version='v1',
        global_crops_scale = (0.9, 1.), 
        global_crop_size = (480, 640),
        local_crops_number = 8,
        local_crops_scale = (0.9, 1.), 
        local_crop_size = (480, 640),
        preprocess_func = lambda x:x.mean(dim=(-1, -2)),
        len_train_loader = 20,
        epochs = 20,
        return_layer='layer4'
    )

    model = torchvision.models.resnet18()
    images = torch.randn(15, 3, 480, 640)
    dino_model = DINOWrapper(model, dino_cfgs)  # using DINOv2 now

    # training-time
    original_output, dino_loss = dino_model(images, epoch=0)
    print(original_output)
    print(dino_loss)

    dino_model.ema_update(it=0)
    
    # testing-time
    original_output = dino_model(images, mode='val')
    # breakpoint()
    print(original_output)