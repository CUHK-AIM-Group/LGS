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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, TV_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from os import makedirs
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams, ModelHiddenParams1
from utils.graphics_utils import getWorld2View2

from icecream import ic 
import random
import copy
import json
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from prune import prune_list, calculate_v_imp_score



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)
        

to_tensor = lambda x: x.to("cuda") if isinstance(
    x, torch.Tensor) else torch.Tensor(x).to("cuda")
img2mse = lambda x, y: torch.mean((x - y)**2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(to_tensor([10.]))


def prepare_output_and_logger(args):    
    if not args.model_path:
        unique_str = args.expname
        args.model_path = os.path.join("./output/", unique_str)
    
    args.model_path += args.expname
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, stage="fine"):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)


def distill_training(
        dataset, hyper, opt, pipe, 
        args
    ):

    testing_iterations, saving_iterations = args.test_iterations, args.save_iterations
    checkpoint_iterations, checkpoint, debug_from = args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    new_max_sh = args.new_max_sh
    distill_feature = args.distill_feature
    first_iter = 0
    old_sh_degree = dataset.sh_degree
    dataset.sh_degree = new_max_sh

    tb_writer = prepare_output_and_logger(args)    
    with torch.no_grad():
        teacher_gaussians = GaussianModel(old_sh_degree, hyper, distill_feature=distill_feature) 
        # teacher_gaussians.training_setup(opt)
    dataset.model_path = args.model_path

    student_gaussians = GaussianModel(old_sh_degree, hyper, distill_feature=distill_feature)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # record the pruned gaussians; True means the corresponding gaussian has been pruned; modified by Piang
    prune_mask = None
    if checkpoint:
        (teacher_model_params, _) = torch.load(args.teacher_model)
        (model_params, first_iter) = torch.load(checkpoint)
        teacher_gaussians.restore(teacher_model_params, copy.deepcopy(opt))
        teacher_gaussians.load_model(args.deformatioin_model_path)

        student_gaussians.restore(model_params, opt)
        student_gaussians.load_model(args.deformatioin_model_path)

        student_scene = Scene(dataset, student_gaussians, load=False)
        if args.prune_sh:
            student_gaussians.max_sh_degree = new_max_sh
            student_gaussians.onedownSHdegree()
            # student_gaussians.onedownSHdegree() # modified by Piang
        if args.prune:
            gaussian_list, imp_list, opacity_list, delta_scale_list = prune_list(student_gaussians, student_scene, pipe, background, distill_feature=distill_feature)
            i = 0

            deform_sort, _ = torch.sort(delta_scale_list, dim=0)
            index_nth_percentile = int(args.prune_threshold * (deform_sort.shape[0] - 1))
            value_nth_percentile = deform_sort[index_nth_percentile]
            deformation_point = (delta_scale_list >= value_nth_percentile).squeeze()

            opacity_list = gaussian_list * opacity_list.reshape(-1)
            delta_scale_list = gaussian_list * delta_scale_list.reshape(-1)

            v_list = calculate_v_imp_score(student_gaussians, imp_list, args.v_pow)

            # modified by piang
            v_list[deformation_point] = v_list[deformation_point] * delta_scale_list[deformation_point]
            v_list[~deformation_point] = v_list[~deformation_point] * opacity_list[~deformation_point]

            prune_mask = student_gaussians.prune_gaussians_with_deform(
                (args.prune_decay**i) * args.prune_percent, v_list, deformation_point
            )
            
        if args.prune_deform:
            hyper.kplanes_config = hyper.new_kplanes_config
            student_gaussians.change_deformation2(hyper)
        student_scene.gaussians = student_gaussians
        
    student_gaussians.training_setup(opt)
    if (not args.enable_covariance):
        student_gaussians._scaling.requires_grad = False
        student_gaussians._rotation.requires_grad = False
    if (not args.enable_opacity):
        student_gaussians._opacity.requires_grad = False
        
    teacher_gaussians.optimizer = None
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    final_iter = opt.iterations + 1
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")

    first_iter += 1

    for iteration in range(first_iter, final_iter + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, student_gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        student_gaussians.update_learning_rate(iteration - first_iter)

        # Every 500 iterations step in scheduler
        if iteration % 500 == 0:
            # student_gaussians.oneupSHdegree()
            # student_gaussians.scheduler.step()
            pass

        if not viewpoint_stack:
            viewpoint_stack = copy.copy(student_scene.getTrainCameras())
        viewpoint_cam_org = viewpoint_stack[randint(0, len(viewpoint_stack)-1)]
        viewpoint_cam = copy.deepcopy(viewpoint_cam_org)

        student_render_pkg = render(viewpoint_cam, student_gaussians, pipe, background, distill_feature=distill_feature)
        student_image = student_render_pkg["render"]
        mask = viewpoint_cam.mask.cuda()
        teacher_render_pkg = render(viewpoint_cam, teacher_gaussians, pipe, background, distill_feature=distill_feature)
        teacher_image = teacher_render_pkg["render"].detach()

        Ll1 = l2_loss(student_image * mask, teacher_image * mask)
        
        loss = Ll1
        if args.distill_feature:
            student_feature = student_render_pkg["deformation_feature"]
            teacher_feature = teacher_render_pkg["deformation_feature"].detach()
            # notice: should be prune_mask1, otherwise, the value will be modified in different iterations
            prune_mask1 = ~prune_mask
            teacher_feature = teacher_feature[prune_mask1]
            L_feature = l2_loss(student_feature, teacher_feature)
            loss += L_feature * 0.1

        if args.gt:
            gt_image = viewpoint_cam.original_image.cuda().float()
            L_gt = l2_loss(student_image * mask, gt_image * mask)
            loss += L_gt

        if opt.lambda_dssim != 0:
            ssim_loss = ssim(teacher_image * mask, student_image * mask)
            ssim_loss1 = ssim(gt_image * mask, student_image * mask)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
            loss += opt.lambda_dssim * (1.0-ssim_loss1)

        loss.backward()
        iter_end.record()

        if iteration < opt.iterations:
                student_gaussians.optimizer.step()
                student_gaussians.optimizer.zero_grad(set_to_none = True)

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                ic(student_gaussians._features_rest.detach().shape)
                student_scene.save(iteration, stage="fine")

                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(student_scene.model_path):
                    os.makedirs(student_scene.model_path)
                torch.save((student_gaussians.capture(), iteration), student_scene.model_path + "/chkpnt" + str(iteration) + ".pth")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[35_001, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 3000, 4000, 5000, 6000, 7000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2000, 3000])
    parser.add_argument("--start_checkpoint", type=str, default = "/path/to/teacher/model/")
    parser.add_argument("--new_max_sh", type=int, default = 2)
    parser.add_argument("--augmented_view", action="store_true")
    parser.add_argument("--enable_covariance", action="store_true")
    parser.add_argument("--enable_opacity", action="store_true")
    parser.add_argument("--opacity_prune", type=float, default=0)
    parser.add_argument("--deformatioin_model_path", type=str, default="/path/to/teacher/model/")
    parser.add_argument("--teacher_model", type=str, default="/path/to/teacher/model/")
    parser.add_argument("--expname", type=str, default="/name/of/output/file")
    parser.add_argument("--configs", type=str, default = "/path/to/config")

    parser.add_argument(
        "--prune_iterations", nargs="+", type=int, default=[16_000, 24_000]
    )
    parser.add_argument("--prune_percent", type=float, default=0.6)
    parser.add_argument("--v_pow", type=float, default=0.1)
    parser.add_argument("--prune_decay", type=float, default=0.6)
    parser.add_argument("--prune", action='store_true')
    parser.add_argument("--prune_threshold", type=float, default=0.6)
    parser.add_argument("--prune_sh", action='store_true')
    parser.add_argument("--prune_deform", action='store_true')
    parser.add_argument("--distill_feature", action='store_true')
    parser.add_argument("--gt", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.checkpoint_iterations = args.save_iterations

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    distill_training(
        lp.extract(args), 
        hp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args, 
    )

    # All done
    print("\nTraining complete.")



