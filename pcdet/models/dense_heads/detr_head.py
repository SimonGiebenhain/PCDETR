import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from .detr.position_encoding import get_sine_embed
from .detr_head_template import DetrHeadTemplate
from .detr.transformer import build_transformer
from .detr.matcher import build_matcher
from .detr.detr import SetCriterion

#TODO SIMON: print size of 2d feature map
#TODO SIMON: print allota stuff, in order to debug more efficiently on CCU

DEFAULT_NUM_QUERIES = 100
DEFAULT_AUX_LOSS = True
DEFAULT_NUM_POS_EMBEDDING = 64
DEFAULT_TEMPERATURE = 10000
#more default params in transformer, matcher module

class DetrHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__()
        num_class = num_class + 1 #this has to do with the implementation of DETR, as class idx 0 is used for non-obj
        self.num_queries = model_cfg.DETR['NUM_QUERIES']
        self.transformer = build_transformer(model_cfg.TRANSFORMER)
        hidden_dim = self.transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_class + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 7, 3)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim) #stores 'self.num_queries' vectors of dim 'hidden_dim'
        self.input_proj = lambda x: x #nn.Conv2d(input_channels, hidden_dim, kernel_size=1) #reduce num channles of feature map
        #self.backbone = backbone
        self.aux_loss = model_cfg.get('USE_AUX_LOSS', DEFAULT_AUX_LOSS)
        self.num_pos_embedding = model_cfg.get('NUM_POS_EMBEDDING', DEFAULT_NUM_POS_EMBEDDING)

        self.forward_ret_dict = {}

        self.matcher = build_matcher(model_cfg.MATCHER)

        self.weight_dict = {'loss_ce': model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['CE_COEF'],
                       'loss_bbox': model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['BBOX_COEF']}
                       #'loss_giou': model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['giou_loss_coef']}

        # TODO FARI: this is a hack
        if self.aux_loss:
            aux_weight_dict = {}
            for i in range(model_cfg.TRANSFORMER.NUM_DECODER_LAYERS - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']

        self.criterion = SetCriterion(num_class, matcher=self.matcher, weight_dict=self.weight_dict,
                                 eos_coef=model_cfg.LOSS_CONFIG['EOS_COEF'], losses=losses)
        self.criterion.cuda() #TODO do this properly

        #TODO do i need to push all model components to device?

        #TODO SIMON: what is direction loss
        #if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
        #    self.conv_dir_cls = nn.Conv2d(
        #        input_channels,
        #        self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
        #        kernel_size=1
        #    )
        #else:
        #    self.conv_dir_cls = None


    #TODO: for now I will ignor mask and remove nestedTensor
    def forward(self, data_dict):
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
        print('Spatial Features 2D:')
        print(spatial_features_2d.shape)

        [nb, nx, ny] = spatial_features_2d.shape
        # generate positional embedding
        pos = get_sine_embed(nb, nx, ny, self.num_pos_embedding, DEFAULT_TEMPERATURE, spatial_features_2d.device)

        # flattening of spatial_features_2d happens inside transformer
        hs = self.transformer(self.input_proj(spatial_features_2d), self.query_embed.weight, pos)[0]
        print('Tansformer Output')
        print(hs.shape)


        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs) #TODO what should I scale this to

        # [-1] index extracts reults after final transformer-decoder layer
        self.forward_ret_dict['cls_preds'] = outputs_class[-1]
        self.forward_ret_dict['box_preds'] = outputs_coord[-1]
        if self.aux_loss:
            self.forward_ret_dict['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            self.forward_ret_dict.update(data_dict['gt_boxes'])
            print('Gt bpxes:')
            print(data_dict['gt_boxes'].shape)

        #TODO when not training: generate bboxes here and add to data_dict, but check out AnchorHeadSingle as template

        return data_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def get_loss(self):
        loss_dict = self.criterion(self.forward_ret_dict)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        tb_dict = {}
        for (k, v) in loss_dict:
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