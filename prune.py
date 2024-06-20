#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import torch
from random import randint
from gaussian_renderer import render, count_render, count_deformation_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getWorld2View2
from icecream import ic
import random
import copy
import gc
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict


def calculate_v_imp_score(gaussians, imp_list, v_pow):
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method.
    :param imp_list: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    # Calculate the volume of each Gaussian component
    gaussians._scaling = torch.nn.Parameter(gaussians._scaling)
    volume = torch.prod(gaussians.get_scaling, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    # modified by piang
    # v_list = v_list * imp_list
    return v_list



# modified by piang
def prune_list(gaussians, scene, pipe, background, distill_feature=False):
    viewpoint_stack = scene.getTrainCameras()
    gaussian_list, imp_list = None, None
    viewpoint_cam = viewpoint_stack[0]
    # render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
    render_pkg = count_deformation_render(viewpoint_cam, gaussians, pipe, background, distill_feature=distill_feature)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )
    opacity_list = render_pkg['opacity']
    delta_scale_list = render_pkg['delta_scale']
    # ic(dataset.model_path)
    for iteration in range(1, len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        viewpoint_cam = viewpoint_stack[iteration]
        # render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        render_pkg = count_deformation_render(viewpoint_cam, gaussians, pipe, background, distill_feature=distill_feature)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        opacity_list1 = render_pkg['opacity']
        gaussian_list += gaussians_count
        imp_list += important_score
        delta_scale_list += render_pkg['delta_scale']
        gc.collect()
    return gaussian_list, imp_list, opacity_list, delta_scale_list

