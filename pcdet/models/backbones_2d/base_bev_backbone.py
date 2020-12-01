import numpy as np
import torch
import torch.nn as nn
import math
import re


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            self.keeps_resolution = True
            for i in range(len(layer_strides)):
                if layer_strides[i] != 1:
                    self.keeps_resolution = False
                    break
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        if len(self.deblocks) > 0:
            downsamlpe_factor = 1
            for i in range(len(layer_strides)):
                downsamlpe_factor *= layer_strides[i]
            upsample_factor = 1
            for i in range(len(upsample_strides)):
                upsample_factor *= upsample_strides[i]
            if downsamlpe_factor == upsample_factor:
                self.keeps_resolution = True

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        if self.keeps_resolution:
            self.num_bev_features = c_in
        else:
            self.num_bev_features = num_filters[-1]

        if not self.keeps_resolution:
            for s in layer_strides[1:]:
                self.grid_size[0] = int(self.grid_size[0] / s)
                self.grid_size[1] = int(self.grid_size[1] / s)

        if 'CKPT_PATH' in model_cfg:
            model_state = torch.load(model_cfg.CKPT_PATH)['model_state']
            model_state = {k[k.index('.')+1:]: v for (k, v) in model_state.items() if "backbone" in k}
            print(model_state.keys())
            self.load_state_dict(model_state)
            for params in self.parameters():
                params.requires_grad = False



    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            elif self.keeps_resolution:
                ups.append(x)

        if self.keeps_resolution:
            if len(ups) > 1:
                x = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x = ups[0]

            if len(self.deblocks) > len(self.blocks):
                x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
