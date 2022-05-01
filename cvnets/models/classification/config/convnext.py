
from typing import Dict


def get_configuration(opts) -> Dict:
    convnext_config = dict()
    
    convnext_config["layer2"] = {
        "num_blocks": 3,
        "channels": 42,
        "kernel_size": 3,
        "expan_ratio": 4,
    }
    convnext_config["layer3"] = {
        "num_blocks": 3,
        "channels": 84,
        "kernel_size": 5,
        "expan_ratio": 4,
    }
    convnext_config["layer4"] = {
        "num_blocks": 9,
        "channels": 162,
        "kernel_size": 7,
        "expan_ratio": 4,
    }
    convnext_config["layer5"] = {
        "num_blocks": 3,
        "channels": 336,
        "kernel_size": 9,
        "expan_ratio": 4,
    }
    
    return convnext_config
