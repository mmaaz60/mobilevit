
from typing import Dict


def get_configuration(opts) -> Dict:
    convnext_config = dict()

    convnext_config["conv1"] = {
        "out_channels": 42,
        "kernel_size": 4,
        "stride": 4,
    }
    convnext_config["layer2"] = {
        "num_blocks": 3,
        "out_channels": 42,
        "kernel_size": 3,
        "expan_ratio": 4,
    }
    convnext_config["layer3"] = {
        "num_blocks": 3,
        "out_channels": 84,
        "kernel_size": 5,
        "expan_ratio": 4,
    }
    convnext_config["layer4"] = {
        "num_blocks": 9,
        "out_channels": 162,
        "kernel_size": 7,
        "expan_ratio": 4,
    }
    convnext_config["layer5"] = {
        "num_blocks": 3,
        "out_channels": 336,
        "kernel_size": 9,
        "expan_ratio": 4,
    }
    
    return convnext_config
