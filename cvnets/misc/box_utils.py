#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import Tensor


"""
    This file implements different conversion functions to implement SSD (https://arxiv.org/pdf/1512.02325.pdf).
    Equations are written inside each function for brevity.
"""


def convert_locations_to_boxes(pred_locations: Tensor,
                               anchor_boxes: Tensor,
                               center_variance: float,
                               size_variance: float):
    """
    This is an inverse of convert_boxes_to_locations function (or Eq.(2) in paper https://arxiv.org/pdf/1512.02325.pdf)
    :param pred_locations: predicted locations from detector
    :param anchor_boxes: prior boxes in center form
    :param center_variance: variance value for centers (c_x and c_y)
    :param size_variance: variance value for size (height and width)
    :return: predicted boxes in center form
    """
    # priors can have one dimension less.
    if anchor_boxes.dim() + 1 == pred_locations.dim():
        anchor_boxes = anchor_boxes.unsqueeze(0)

    # T_w = log(g_w/d_w) / size_variance ==> g_w = exp(T_w * size_variance) * d_w
    # T_h = log(g_h/d_h) / size_variance ==> g_h = exp(T_h * size_variance) * d_h
    pred_size = torch.exp(pred_locations[..., 2:] * size_variance) * anchor_boxes[..., 2:]
    # T_cx = ((g_cx - d_cx) / d_w) / center_variance ==> g_cx = ((T_cx * center_variance) * d_w) + d_cx
    # T_cy = ((g_cy - d_cy) / d_w) / center_variance ==> g_cy = ((T_cy * center_variance) * d_h) + d_cy
    pred_center = (pred_locations[..., :2] * center_variance * anchor_boxes[..., 2:]) + anchor_boxes[..., :2]

    return torch.cat((pred_center, pred_size), dim=-1)


def convert_boxes_to_locations(gt_boxes: Tensor,
                               prior_boxes: Tensor,
                               center_variance: float,
                               size_variance: float):
    """
    This function implements Eq.(2) in paper https://arxiv.org/pdf/1512.02325.pdf
    :param gt_boxes: Ground truth boxes in center form [g_cx, g_cy, g_w, g_h]
    :param prior_boxes: Prior boxes in center form [d_cx, d_cy, d_w, d_h]
    :param center_variance: variance value for centers (c_x and c_y)
    :param size_variance: variance value for size (height and width)
    :return: boxes for regression
    """

    # T_cx = ((g_cx - d_cx) / d_w) / center_variance; Center vairance is nothing but normalization
    # T_cy = ((g_cy - d_cy) / d_h) / center_variance
    # T_w = log(g_w/d_w) / size_variance and T_h = log(g_h/d_h) / size_varianc

    # priors can have one dimension less
    if prior_boxes.dim() + 1 == gt_boxes.dim():
        prior_boxes = prior_boxes.unsqueeze(0)

    target_centers = ((gt_boxes[..., :2] - prior_boxes[..., :2]) / prior_boxes[..., 2:])/center_variance
    target_size = torch.log(gt_boxes[..., 2:] / prior_boxes[..., 2:]) / size_variance
    return torch.cat((target_centers, target_size), dim=-1)


def center_form_to_corner_form(boxes: torch.Tensor) -> torch.Tensor:
    """

    :param boxes: boxes in center form (c_x, c_y, delta_w, delta_h) [N, 4]
    :return:  boxes in corner form (x, y, w, h) [N, 4]
    """
    # [c_x, c_y, top_left_x, top_left_y] --> [x, y, w, h]
    # x = c_x - (delta_w * 0.5), y = c_y - (delta_h * 0.5)
    # w = c_x + (delta_w * 0.5), h = c_y + (delta_h * 0.5)
    return torch.cat(
        (
            boxes[..., :2] - (boxes[..., 2:] * 0.5),
            boxes[..., :2] + (boxes[..., 2:] * 0.5)
        ), dim=-1)


def corner_form_to_center_form(boxes: torch.Tensor) -> torch.Tensor:
    """
    :param boxes: boxes in corner form (x, y, w, h) [N, 4]
    :return: boxes in center form (c_x, c_y, left, right) [N, 4]
    """
    # [x, y, w, h] --> [c_x, c_y, delta_w, delta_h]
    # c_x = ( x + w ) * 0.5, c_y = (y + h) * 0.5
    # delta_w = w - x, delta_h = h - y
    return torch.cat(
        (
            (boxes[..., :2] + boxes[..., 2:]) * 0.5,
            boxes[..., 2:] - boxes[..., :2]
        ),
        dim=-1
    )
