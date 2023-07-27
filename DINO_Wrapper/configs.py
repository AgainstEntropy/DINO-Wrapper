from typing import Optional
import yaml


SHARED_DEFAULT_CONFIGS = {
    # Model
    'out_dim': None,
    # Multi-crop parameters
    'global_crops_number': 2,
    'global_crop_size': 224,
    'local_crops_number': 8,
    'local_crop_size': 96,
    # Temperature teacher parameters
    'warmup_teacher_temp': 0.04,
    # Train
    'len_train_loader': None,
    'epochs': 100,
    # Others
    'preprocess_func': None,
    # Use-Lyaers
    'return_layer': None,
}

V1_DEFAULT_CONFIGS = {
    'version': 'v1',
    # Multi-crop parameters
    'global_crops_scale': (0.4, 1.), 
    'local_crops_scale': (0.05, 0.4), 
    # Model parameters
    'momentum_teacher': 0.996,
    # Temperature teacher parameters
    'teacher_temp': 0.04,
    'warmup_teacher_temp_epochs': 0,
}

V2_DEFAULT_CONFIGS = {
    'version': 'v2',
    # Multi-crop parameters
    'global_crops_scale': (0.32, 1.), 
    'local_crops_scale': (0.05, 0.32), 
    # Model parameters
    'momentum_teacher': 0.992,
    # Temperature teacher parameters
    'teacher_temp': 0.07,
}


class DINOConf:
    def __init__(self, version: Optional[str], **kwargs):
        default_configs = SHARED_DEFAULT_CONFIGS
        if version == 'v1' or version is None:
            default_configs.update(V1_DEFAULT_CONFIGS)
        elif version == 'v2':
            default_configs.update(V2_DEFAULT_CONFIGS)
        else:
            raise ValueError(f"Version {version} not supported. Only support v1 and v2.")

        self.known_keys = set(default_configs.keys())
        for k, v in default_configs.items():
            setattr(self, k, v)

        self.merge(kwargs)

    def load(self, filename: str):
        with open(filename, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        self.merge(config)

    def merge(self, new_configs: dict):
        for k in self.known_keys.intersection(set(new_configs.keys())):
            setattr(self, k, new_configs.pop(k))

        if len(new_configs) != 0:
            print(f"The following unknown configurations are not used:\n"
                   "{list(new_configs.keys())}\n"
                   "Please remove them next time.\n")
            