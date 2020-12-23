import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from .detr.position_encoding import get_sine_embed
from .detr_head_template import DetrHeadTemplate
from .detr.transformer import build_transformer
from .detr.matcher import build_matcher
from .detr.detr import SetCriterion

DEFAULT_NUM_QUERIES = 100
DEFAULT_AUX_LOSS = True
DEFAULT_NUM_POS_EMBEDDING = 64
DEFAULT_TEMPERATURE = 10000
#more default params in transformer, matcher module

'''
The code for this was more or less copied from the DETR github
I only adjusted where necessary
Where I wrote "TODO FAIR" it means that the "TODO" is from the FAIR(Facebook AI Research) github
'''

class DetrHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__()
        self.num_queries = model_cfg.DETR['NUM_QUERIES']
        self.transformer = build_transformer(model_cfg.TRANSFORMER)
        hidden_dim = self.transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_class + 1)
        # 7-dim output since we predict (x,y,z,w,h,d,angle)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 7, 3)
        # I used a tanh() activation and use this scaling to map from [-1, 1] to suitable domain for every dimension
        # TODO: I could have done this better, e.g. not sure whether the maximum width is 5, also width cannot be negative etc.
        self.scale = torch.tensor([69.12, 39.68, 3.0, 5.0, 5.0, 2.0, 2.0]).cuda()
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim) #stores 'self.num_queries' vectors of dim 'hidden_dim'
        self.input_proj = nn.Conv2d(input_channels, hidden_dim, kernel_size=1) #reduce num channles of feature map
        self.aux_loss = model_cfg.get('USE_AUX_LOSS', DEFAULT_AUX_LOSS)
        self.num_pos_embedding = model_cfg.get('NUM_POS_EMBEDDING', DEFAULT_NUM_POS_EMBEDDING)

        # This dict will hold all necessary outputs and intermediate outputs to compute the los later on
        self.forward_ret_dict = {}

        self.matcher = build_matcher(model_cfg.MATCHER)

        # weights for the losses, as I said I did not implement the giou loss
        self.weight_dict = {'loss_ce': model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['CE_COEF'],
                            'loss_bbox': model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['BBOX_COEF']}
                            #'loss_giou': model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['giou_loss_coef']}


        # TODO FAIR: this is a hack
        if self.aux_loss:
            aux_weight_dict = {}
            for i in range(model_cfg.TRANSFORMER.NUM_DECODER_LAYERS - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']

        # This module computes the bipartite matching an computes the loss
        self.criterion = SetCriterion(num_class, matcher=self.matcher, weight_dict=self.weight_dict,
                                 eos_coef=model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['EOS_COEF'], losses=losses)
        self.criterion.cuda()

        #TODO: what is direction loss, it is used in the PointPillar model, but I don't know what it is
        #if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
        #    self.conv_dir_cls = nn.Conv2d(
        #        input_channels,
        #        self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
        #        kernel_size=1
        #    )
        #else:
        #    self.conv_dir_cls = None


    # I ignoreg mask and therefore removed nestedTensor
    # This was not necessary as the all point clouds are preprocessed to the same size
    # The original DETR implementation needed masks and nestedTensors to accommodate for different sized images
    def forward(self, data_dict):
        #TODO write new comment
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        spatial_features_2d = data_dict['spatial_features_2d']

        [nb, _, nx, ny] = spatial_features_2d.shape
        # generate positional embedding
        # I re-implemented this because it was coded for masks
        # TODO test if positional embedding works correctly
        pos = get_sine_embed(nb, nx, ny, self.num_pos_embedding, DEFAULT_TEMPERATURE, spatial_features_2d.device)

        # flattening of spatial_features_2d happens inside transformer
        hs = self.transformer(self.input_proj(spatial_features_2d), self.query_embed.weight, pos)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).tanh()
        # TODO come up with better scaling
        outputs_coord = outputs_coord * self.scale

        # [-1] index extracts results after final transformer-decoder layer
        self.forward_ret_dict['cls_preds'] = outputs_class[-1]
        self.forward_ret_dict['box_preds'] = outputs_coord[-1]
        # also store intermediate outcomes for auxiliary losses
        if self.aux_loss:
            self.forward_ret_dict['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']

        #TODO when not training: implement what AnchorHeadSingle does here

        return data_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'cls_preds': a, 'box_preds': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def get_loss(self):
        loss_dict = self.criterion(self.forward_ret_dict)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        tb_dict = {}
        for (k, v) in loss_dict.items():
            tb_dict[k] = v.item()
        tb_dict['total_loss'] = losses.item()
        return losses, tb_dict


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x