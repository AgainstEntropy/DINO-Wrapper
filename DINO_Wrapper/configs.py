from typing import Any
from omegaconf import OmegaConf
import yaml


DEFAULT_CONFIGS = {
    'version': 'v1',
    'local_crops_number': 2,
    'global_crops_scale': (0.9, 1.), 
    'global_crop_size': (480, 640),
    'local_crops_number': 8,
    'local_crops_scale': (0.6, 0.7), 
    'local_crop_size': (240, 320),
    'preprocess_func': None,
}


class DinoConf:
    def __init__(self, **kwargs):
        self.conf = OmegaConf.create(DEFAULT_CONFIGS)
        self.merge(kwargs)

        if self.conf.version not in ['v1', 'v2']:
            raise ValueError(f"Version {self.conf.version} not supported")

    def load(self, filename: str):
        with open(filename, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        self.merge(config)

    def merge(self, new_configs: dict):
        for k in DEFAULT_CONFIGS.keys():
            setattr(self.conf, k, new_configs.pop(k))

        if len(new_configs.keys()) != 0:
            print(f"""
                    The following unknown configurations are not used:\n
                    {list(new_configs.keys())}\n
                    Please remove them next time.\n
                  """)
            
    def __getattribute__(self, __name: str) -> Any:
        return self.conf.get(__name, None)