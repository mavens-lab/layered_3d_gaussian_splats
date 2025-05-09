#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from lpipsPyTorch import lpips

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.logger_utils import training_report, prepare_output_and_logger

import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# from prune_train import prepare_output_and_logger, training_report
from icecream import ic
from os import makedirs
from prune import prune_list, calculate_v_imp_score
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
import csv
import numpy as np


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
# fix seed
torch.manual_seed(0)


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):
    # def set_requires_grad(tensor, requires_grad):
    #     """Returns a new tensor with the specified requires_grad setting."""
    #     return tensor.detach().clone().requires_grad_(requires_grad)
    def sample(sample_size,gaussians,viewspace_point_tensor):
        # pre_trained_size = len(gaussians.pre_trained_xyz)
        gaussian_small = deepcopy(gaussians)
        viewspace_point_tensor_small = viewspace_point_tensor
        gaussian_small._xyz = gaussian_small._xyz[: sample_size]
        gaussian_small._features_dc = gaussian_small._features_dc[: sample_size]
        gaussian_small._features_rest = gaussian_small._features_rest[: sample_size]
        gaussian_small._opacity = gaussian_small._opacity[: sample_size]
        gaussian_small._scaling = gaussian_small._scaling[: sample_size]
        gaussian_small._rotation = gaussian_small._rotation[: sample_size]
        viewspace_point_tensor_small = 0.0
        return gaussian_small,viewspace_point_tensor_small

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    freeze_size = 90000
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    opt.iterations = 100_000
    opt.densify_until_iter = 50_000
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.97)
    # gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=1)
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            gaussians.scheduler.step()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg_big = render(viewpoint_cam, gaussians, pipe, background)
        image_big, viewspace_point_tensor_big, visibility_filter_big, radii_big = (
            render_pkg_big["render"],
            render_pkg_big["viewspace_points"],
            render_pkg_big["visibility_filter"],
            render_pkg_big["radii"],
        )
        gaussian_small = sample(90000,gaussians,viewspace_point_tensor_big)
        render_pkg_small = render(viewpoint_cam, gaussian_small, pipe, background)
        image_small, viewspace_point_tensor_small, visibility_filter_small, radii_small = (
            render_pkg_small["render"],
            render_pkg_small["viewspace_points"],
            render_pkg_small["visibility_filter"],
            render_pkg_small["radii"],
        )
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1_big = l1_loss(image_big, gt_image)
        Ll1_small = l1_loss(image_small, gt_image)
        Ll1 = (Ll1_big + Ll1_small)/2
        ssim_all = (ssim(image_big, gt_image)+ssim(image_small, gt_image))/2
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim_all
        )
        loss.backward()
        # gaussians,viewspace_point_tensor = freeze(freeze_size,gaussians,viewspace_point_tensor)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # if iteration in saving_iterations:
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )

            # Densification
            if iteration == opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        gaussians.reset_opacity()
                # TODO Add prunning types
                gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                # i = args.prune_iterations.index(iteration)
                v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                # gaussians.prune_gaussians(
                #     (args.prune_decay**i) * args.prune_percent, v_list
                # )

                # scene.gaussians.get_xyz.shape[0]
                # gaussians.prune_gaussians(
                #     args.max_splat_num, v_list
                # )

                if scene.gaussians.get_xyz.shape[0] > args.max_splat_num: 
                    prune_frac = min(1-(args.max_splat_num/scene.gaussians.get_xyz.shape[0]),0.2)

                    gaussians.prune_gaussians(
                        prune_frac, v_list
                    )
            if iteration < opt.densify_until_iter:
                if scene.gaussians.get_xyz.shape[0] > args.max_splat_num and iteration%1000 == 0: 
                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        gaussians.reset_opacity()
                
                    # TODO Add prunning types
                    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                    # i = args.prune_iterations.index(iteration)
                    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                    # gaussians.prune_gaussians(
                    #     (args.prune_decay**i) * args.prune_percent, v_list
                    # )

                    # scene.gaussians.get_xyz.shape[0]
                    # gaussians.prune_gaussians(
                    #     args.max_splat_num, v_list
                    # )
                    if 1-(args.max_splat_num/scene.gaussians.get_xyz.shape[0])>0.01:
                        opt.densify_grad_threshold = opt.densify_grad_threshold*1.05
                        print(opt.densify_grad_threshold)
                        if tb_writer:
                            tb_writer.add_scalar("density/densify_grad_threshold", opt.densify_grad_threshold, iteration)
                    prune_frac = min(1-(args.max_splat_num/scene.gaussians.get_xyz.shape[0]),0.2)
                    gaussians.prune_gaussians(
                        prune_frac, v_list
                    )
                    gaussians.prune_gaussians(
                        1-(args.max_splat_num/scene.gaussians.get_xyz.shape[0]), v_list
                    )
                else:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter_big] = torch.max(
                        gaussians.max_radii2D[visibility_filter_big], radii_big[visibility_filter_big]
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor_big, visibility_filter_big
                    )

                    if (
                        iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                    ):
                        size_threshold = (
                            20 if iteration > opt.opacity_reset_interval else None
                        )
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                        )

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(scene.model_path):
                    os.makedirs(scene.model_path)
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
                if iteration == checkpoint_iterations[-1]:
                    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                    np.savez(os.path.join(scene.model_path,"imp_score"), v_list.cpu().detach().numpy()) 


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[500,7_000, 10_000,15_000,20_000,30_000,40_000,44_000,46_000,48_000,49_000,50_000],
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[500,7_000, 10_000,15_000,20_000,30_000,40_000,44_000,46_000,48_000,49_000,50_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[500,7_000, 10_000,15_000,20_000,30_000,40_000,44_000,46_000,48_000,49_000,50_000]
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # parser.add_argument(
    #     "--prune_iterations", nargs="+", type=int, default=[40000,42000,45000,47000]
    # )
    # parser.add_argument("--prune_percent", type=float, default=0.5)
    parser.add_argument("--v_pow", type=float, default=0.1)
    # parser.add_argument("--prune_decay", type=float, default=0.8)
    parser.add_argument("--max_splat_num", type=int, default=180_000)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    # print(args.max_splat_num)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    # All done
    print("\nTraining complete.")
