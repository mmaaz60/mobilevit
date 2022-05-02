from torch import nn
from typing import Tuple, Dict

from . import register_cls_models
from .base_cls import BaseEncoder
from .config.convnext import get_configuration
from ...layers import ConvLayer, LinearLayer, GlobalPool, Identity, Dropout
from ...modules import ConvNeXtBlock


@register_cls_models("convnext")
class ConvNext(BaseEncoder):
    def __init__(self, opts, *args, **kwargs) -> None:
        image_channels = 3
        input_channels = 64
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.2)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        cfg = get_configuration(opts=opts)

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the ResNet backbone
        # We allow that using output_stride and replace_stride_with_dilation arguments
        output_stride = kwargs.get("output_stride", None)
        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True

        super(ConvNext, self).__init__()
        self.dilation = 1
        self.model_conf_dict = dict()

        # Make stem -> conv1
        layer_config = cfg['conv1']
        stem_channels = layer_config.get("out_channels", 42)
        kernel_size = layer_config.get("kernel_size", 4)
        stride = layer_config.get("stride", 4)
        self.conv_1 = ConvLayer(opts=opts, in_channels=image_channels, out_channels=stem_channels,
                                kernel_size=kernel_size, stride=stride, use_norm=True, use_act=False)
        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': stem_channels}

        # layer1 -> Identity (Later can be used to modify the stem)
        input_channels = stem_channels
        self.layer_1 = Identity()
        self.model_conf_dict['layer1'] = {'in': input_channels, 'out': input_channels}

        # layer2 -> Stage 1 of ConvNeXt + Down sampling
        self.layer_2, out_channels = self._make_layer(opts=opts,
                                                      in_channels=input_channels,
                                                      layer_config=cfg["layer2"],
                                                      layer_name="stage_1",
                                                      downsampling=True
                                                      )
        self.model_conf_dict['layer2'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        # layer3 -> Stage 2 of ConvNeXt + Down sampling
        self.layer_3, out_channels = self._make_layer(opts=opts,
                                                      in_channels=input_channels,
                                                      layer_config=cfg["layer3"],
                                                      layer_name="stage_2",
                                                      downsampling=True
                                                      )
        self.model_conf_dict['layer3'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        # layer4 -> Stage 3 of ConvNeXt + Down sampling
        self.layer_4, out_channels = self._make_layer(opts=opts,
                                                      in_channels=input_channels,
                                                      layer_config=cfg["layer4"],
                                                      dilate=dilate_l4,
                                                      layer_name="stage_3",
                                                      downsampling=True
                                                      )
        self.model_conf_dict['layer4'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        # layer4 -> Stage 4 of ConvNeXt
        self.layer_5, out_channels = self._make_layer(opts=opts,
                                                      in_channels=input_channels,
                                                      layer_config=cfg["layer5"],
                                                      dilate=dilate_l5,
                                                      layer_name="stage_4",
                                                      downsampling=False
                                                      )
        self.model_conf_dict['layer5'] = {'in': input_channels, 'out': out_channels}
        input_channels = out_channels

        # No 1x1 expansion layer is used in ConvNeXt design
        self.conv_1x1_exp = Identity()
        self.model_conf_dict['exp_before_cls'] = {'in': input_channels, 'out': input_channels}

        # Classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False))
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="classifier_dropout", module=Dropout(p=classifier_dropout))
        self.classifier.add_module(name="classifier_fc",
                                   module=LinearLayer(in_features=input_channels, out_features=num_classes, bias=True))

        self.model_conf_dict['cls'] = {'in': input_channels, 'out': num_classes}

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    def _make_layer(self, opts, layer_config: Dict, in_channels: int,
                    dilate: bool = False, *args, **kwargs) -> Tuple[nn.Sequential, int]:
        num_blocks = layer_config.get("num_blocks")
        out_channels = layer_config.get("out_channels")
        kernel_size = layer_config.get("kernel_size", 7)
        expan_ratio = layer_config.get("expan_ratio", 4)
        downsampling = kwargs["downsampling"]
        layer_name = kwargs["layer_name"]

        stage = nn.Sequential()
        for block_idx in range(1, num_blocks + 1):
            stage.add_module(
                name="block_{}".format(block_idx),
                module=ConvNeXtBlock(opts=opts, in_channels=in_channels, expan_ratio=expan_ratio,
                                     kernel_size=kernel_size, dilation=self.dilation)
            )
        if downsampling:
            stage.add_module(name=f"downsample_{layer_name}",
                             module=ConvLayer(opts=opts, in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=2, stride=2, use_norm=True, use_act=False))

        return stage, out_channels
