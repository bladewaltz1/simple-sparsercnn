# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import Conv2d
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class Decoder(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        box_pooler = self._init_box_pooler(cfg, input_shape)
        self.box_pooler = box_pooler

        num_classes = cfg.MODEL.SimpleBaseline.NUM_CLASSES
        num_layer = cfg.MODEL.SimpleBaseline.NUM_LAYERS
        decoder_layer = DecoderLayer(cfg)
        self.decoder_layers = _get_clones(decoder_layer, num_layer)

        self.num_classes = num_classes
        prior_prob = cfg.MODEL.SimpleBaseline.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if p.shape[-1] == self.num_classes:
                nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        in_channels = [input_shape[f].channels for f in in_features]

        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_boxes, queries):
        output = []
        bs = len(features[0])
        boxes = init_boxes
        queries = queries[None].repeat(bs, 1, 1)

        for layer in self.decoder_layers:
            class_logits, pred_boxes, queries = layer(
                features, boxes, queries, self.box_pooler
            )
            output.append({
                'pred_logits': class_logits, 'pred_boxes': pred_boxes
            })
            boxes = pred_boxes.detach()

        return output


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.MODEL.SimpleBaseline.HIDDEN_DIM
        d_feedforward = cfg.MODEL.SimpleBaseline.DIM_FEEDFORWARD
        nhead = cfg.MODEL.SimpleBaseline.NUM_HEADS
        dropout = cfg.MODEL.SimpleBaseline.DROPOUT
        seq_len = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ** 2

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_sim = MultiheadSimilarity(d_model, nhead, seq_len)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        num_cls = cfg.MODEL.SimpleBaseline.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model, False),
                    nn.LayerNorm(d_model),
                    nn.ReLU(inplace=True)
                )
            )
        self.cls_module = nn.ModuleList(cls_module)

        num_reg = cfg.MODEL.SimpleBaseline.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model, False),
                    nn.LayerNorm(d_model),
                    nn.ReLU(inplace=True)
                )
            )
        self.reg_module = nn.ModuleList(reg_module)

        if not cfg.MODEL.SimpleBaseline.USE_FPN:
            self.roi_head = Conv2d(2048, d_model, 1, 1, 0, bias=True)
        else:
            self.roi_head = None

        num_classes = cfg.MODEL.SimpleBaseline.NUM_CLASSES
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)

    def forward(self, features, boxes, queries, pooler):
        """
        boxes: (N, nr_boxes, 4)
        queries: (N, nr_boxes, d_model)
        """
        N, nr_boxes, d_model = queries.shape

        query_boxes = list()
        for b in range(N):
            query_boxes.append(Boxes(boxes[b]))
        roi_features = pooler(features, query_boxes)
        if self.roi_head is not None:
            roi_features = self.roi_head(roi_features)
        roi_features = roi_features.view(N * nr_boxes, d_model, -1)
        roi_features = roi_features.permute(2, 0, 1)

        queries = queries.permute(1, 0, 2)
        queries2 = self.self_attn(queries, queries, queries)[0]
        queries = queries + self.dropout(queries2)
        queries = self.norm1(queries)

        queries = queries.permute(1, 0, 2)
        queries = queries.reshape(N * nr_boxes, d_model)
        queries2 = self.cross_sim(queries, roi_features)
        queries = queries + self.dropout(queries2)
        queries = self.norm2(queries)

        queries2 = self.ffn(queries)
        queries = queries + self.dropout(queries2)
        queries = self.norm3(queries)

        cls_feature = queries.clone()
        reg_feature = queries.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)

        pred_logits = self.class_logits(cls_feature).view(N, nr_boxes, -1)
        deltas = self.bboxes_delta(reg_feature)
        pred_boxes = self.apply_deltas(deltas, boxes.view(-1, 4))
        pred_boxes = pred_boxes.view(N, nr_boxes, -1)
        queries = queries.view(N, nr_boxes, d_model)

        return pred_logits, pred_boxes, queries


    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, 4)
            boxes (Tensor): boxes to transform, shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4] / 2.0
        dy = deltas[:, 1::4] / 2.0
        dw = deltas[:, 2::4] / 1.0
        dh = deltas[:, 3::4] / 1.0

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=_DEFAULT_SCALE_CLAMP)
        dh = torch.clamp(dh, max=_DEFAULT_SCALE_CLAMP)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class MultiheadSimilarity(nn.Module):
    def __init__(self, d_model, nhead, seq_len):
        super().__init__()
        self.nhead = nhead
        self.seq_len = seq_len
        self.d_head = d_model // nhead

        self.q_in_proj = nn.Linear(d_model, seq_len * d_model, bias=True)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(seq_len * d_model, d_model, bias=True)

    def forward(self, q, kv):
        bs, d_model = q.shape
        nbs = bs * self.nhead

        q_ = self.q_in_proj(q)
        q_ = q_.contiguous().view(bs, self.seq_len, d_model).transpose(0, 1)
        kv = q_ + kv

        q = self.q_proj(q)
        q = q.contiguous().view(nbs, self.d_head)
        k = self.k_proj(kv)
        k = k.contiguous().view(self.seq_len, nbs, self.d_head)
        q = F.normalize(q, p=2, dim=-1).unsqueeze(-1)
        k = F.normalize(k, p=2, dim=-1).transpose(0, 1)
        similarity = torch.bmm(k, q)

        v = self.v_proj(kv)
        v = v.contiguous().view(self.seq_len, nbs, self.d_head).transpose(0, 1)
        v = (v * similarity).view(bs, self.nhead, self.seq_len, self.d_head)
        output = self.out_proj(v.flatten(1))
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
