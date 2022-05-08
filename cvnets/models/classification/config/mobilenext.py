
from typing import Dict


def get_configuration(opts) -> Dict:
    mobilenext_config = dict()

    mobilenext_config["conv1"] = {
        "out_channels": 48,
        "kernel_size": 4,
        "stride": 4,
    }
    mobilenext_config["layer2"] = {
        "num_blocks": 3,
        "out_channels": 48,
        "kernel_size": 3,
        "expan_ratio": 4,
        "Conv2DTA": False,
        "d2_scales": 2,
        "num_heads": 8,
        "use_pos_emb": False
    }
    mobilenext_config["layer3"] = {
        "num_blocks": 3,
        "out_channels": 96,
        "kernel_size": 5,
        "expan_ratio": 4,
        "Conv2DTA": True,
        "d2_scales": 3,
        "num_heads": 8,
        "use_pos_emb": True
    }
    mobilenext_config["layer4"] = {
        "num_blocks": 9,
        "out_channels": 160,
        "kernel_size": 7,
        "expan_ratio": 4,
        "Conv2DTA": True,
        "d2_scales": 4,
        "num_heads": 8,
        "use_pos_emb": False
    }
    mobilenext_config["layer5"] = {
        "num_blocks": 3,
        "out_channels": 304,
        "kernel_size": 9,
        "expan_ratio": 4,
        "Conv2DTA": True,
        "d2_scales": 5,
        "num_heads": 8,
        "use_pos_emb": False
    }
    
    return mobilenext_config
