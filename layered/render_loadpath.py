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

import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from icecream import ic
import copy

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spherical_sample_path, generate_spiral_path, generate_spherify_path,  gaussian_poses, circular_poses,generate_xyz_path,generate_rt_path
# import stepfun 
import cv2
from PIL import Image

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


# def normalize(x):
#     return x / np.linalg.norm(x)

# def viewmatrix(z, up, pos):
#     vec2 = normalize(z)
#     vec0 = normalize(np.cross(up, vec2))
#     vec1 = normalize(np.cross(vec2, vec0))
#     m = np.stack([vec0, vec1, vec2, pos], 1)
#     return m

# def poses_avg(poses):
#     hwf = poses[0, :3, -1:]
#     center = poses[:, :3, 3].mean(0)
#     vec2 = normalize(poses[:, :3, 2].sum(0))
#     up = poses[:, :3, 1].sum(0)
#     c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
#     return c2w

# def get_focal(camera):
#     focal = camera.FoVx
#     return focal

# def poses_avg_fixed_center(poses):
#     hwf = poses[0, :3, -1:]
#     center = poses[:, :3, 3].mean(0)
#     vec2 = [1, 0, 0]
#     up = [0, 0, 1]
#     c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
#     return c2w

# def focus_point_fn(poses):
#     """Calculate nearest point to all focal axes in poses."""
#     directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
#     m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
#     mt_m = np.transpose(m, [0, 2, 1]) @ m
#     focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
#     return focus_pt






# xy circular 
def render_circular_video(model_path, iteration, views, gaussians, pipeline, background, radius=0.5, n_frames=240): 
    render_path = os.path.join(model_path, 'circular', "ours_{}".format(iteration))
    os.makedirs(render_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    # view = views[0]
    for idx in range(n_frames):
        view = copy.deepcopy(views[13])
        angle = 2 * np.pi * idx / n_frames
        cam = circular_poses(view, radius, angle)
        rendering = render(cam, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    
    out_path = os.path.join(render_path,'video')
    makedirs(out_path, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    gt = view.original_image[0:3, :, :]
    size = (gt.shape[-1],gt.shape[-2])
    fps = float(72)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

    for file_name in sorted(os.listdir(render_path)):
        rgb = np.array(Image.open(os.path.join(render_path,file_name)))
        rgb = rgb.astype('uint8')
        writer.write(rgb[:,:,::-1])
    writer.release()



def render_video(model_path, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join('/home/zxuechen/LightGaussian/OUTPUT', 'video_ellipse', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    # render_path_spiral
    # render_path_spherical
    with open('ellipse_array.txt','w') as f: 
    # np.savetxt('array.txt',np.array(generate_ellipse_path(views,n_frames=600)),delimiter=',',fmt='%f')
        for idx, pose in enumerate(tqdm(generate_ellipse_path(views,n_frames=600), desc="Rendering progress")):
            np.savetxt(f,pose,delimiter=',',fmt='%f')
            f.write('\n')
            view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.camera_center = view.world_view_transform.inverse()[3, :3]
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    out_path = os.path.join(render_path,'video')
    makedirs(out_path, exist_ok=True)
    #fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
    # fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    first_image_path = os.path.join(render_path, '{0:05d}'.format(0) + ".png")
    first_image = Image.open(first_image_path)
    size = (first_image.width, first_image.height)
    fps = float(72)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)
    if not writer.isOpened():
        print("Error: Failed to open video writer.")
    else:
        print('Video writer opened successfully.')
        image_dir = render_path
        for file_name in sorted(os.listdir(render_path)):
            if file_name.endswith('.png'):
                image_path = os.path.join(image_dir, file_name)
                rgb = np.array(Image.open(image_path))
                rgb = rgb.astype('uint8')
                writer.write(rgb[:,:,::-1])
        writer.release()

    render_path = os.path.join('/home/zxuechen/LightGaussian/OUTPUT', 'videospiral', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    # render_path_spiral
    # render_path_spherical
    with open('spiral_array.txt','w') as f: 
    # np.savetxt('array.txt',np.array(generate_ellipse_path(views,n_frames=600)),delimiter=',',fmt='%f')
        for idx, pose in enumerate(tqdm(generate_spiral_path(views,N=600), desc="Rendering progress")):
            np.savetxt(f,pose,delimiter=',',fmt='%f')
            f.write('\n')
            view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.camera_center = view.world_view_transform.inverse()[3, :3]
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    out_path = os.path.join(render_path,'video')
    makedirs(out_path, exist_ok=True)
    #fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
    # fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    first_image_path = os.path.join(render_path, '{0:05d}'.format(0) + ".png")
    first_image = Image.open(first_image_path)
    size = (first_image.width, first_image.height)
    fps = float(72)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)
    if not writer.isOpened():
        print("Error: Failed to open video writer.")
    else:
        print('Video writer opened successfully.')
        image_dir = render_path
        for file_name in sorted(os.listdir(render_path)):
            if file_name.endswith('.png'):
                image_path = os.path.join(image_dir, file_name)
                rgb = np.array(Image.open(image_path))
                rgb = rgb.astype('uint8')
                writer.write(rgb[:,:,::-1])
        writer.release()

    render_path = os.path.join('/home/zxuechen/LightGaussian/OUTPUT', 'videospherical', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    # render_path_spiral
    # render_path_spherical
    with open('spherical_array.txt','w') as f: 
    # np.savetxt('array.txt',np.array(generate_ellipse_path(views,n_frames=600)),delimiter=',',fmt='%f')
        for idx, pose in enumerate(tqdm(generate_spherical_sample_path(views,N=600), desc="Rendering progress")):
            np.savetxt(f,pose,delimiter=',',fmt='%f')
            f.write('\n')
            view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.camera_center = view.world_view_transform.inverse()[3, :3]
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    out_path = os.path.join(render_path,'video')
    makedirs(out_path, exist_ok=True)
    #fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
    # fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    first_image_path = os.path.join(render_path, '{0:05d}'.format(0) + ".png")
    first_image = Image.open(first_image_path)
    size = (first_image.width, first_image.height)
    fps = float(72)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)
    if not writer.isOpened():
        print("Error: Failed to open video writer.")
    else:
        print('Video writer opened successfully.')
        image_dir = render_path
        for file_name in sorted(os.listdir(render_path)):
            if file_name.endswith('.png'):
                image_path = os.path.join(image_dir, file_name)
                rgb = np.array(Image.open(image_path))
                rgb = rgb.astype('uint8')
                writer.write(rgb[:,:,::-1])
        writer.release()

    render_path = os.path.join('/home/zxuechen/LightGaussian/OUTPUT', 'videospherify', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    # render_path_spiral
    # render_path_spherical
    with open('spherify_array.txt','w') as f: 
    # np.savetxt('array.txt',np.array(generate_ellipse_path(views,n_frames=600)),delimiter=',',fmt='%f')
        for idx, pose in enumerate(tqdm(generate_spherify_path(views), desc="Rendering progress")):
            np.savetxt(f,pose,delimiter=',',fmt='%f')
            f.write('\n')
            view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view.camera_center = view.world_view_transform.inverse()[3, :3]
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    out_path = os.path.join(render_path,'video')
    makedirs(out_path, exist_ok=True)
    #fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
    # fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    first_image_path = os.path.join(render_path, '{0:05d}'.format(0) + ".png")
    first_image = Image.open(first_image_path)
    size = (first_image.width, first_image.height)
    fps = float(72)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)
    if not writer.isOpened():
        print("Error: Failed to open video writer.")
    else:
        print('Video writer opened successfully.')
        image_dir = render_path
        for file_name in sorted(os.listdir(render_path)):
            if file_name.endswith('.png'):
                image_path = os.path.join(image_dir, file_name)
                rgb = np.array(Image.open(image_path))
                rgb = rgb.astype('uint8')
                writer.write(rgb[:,:,::-1])
        writer.release()

def render_xyz_trace(model_path, iteration, views, gaussians, pipeline, background,save_video):
    render_path = os.path.join('/nfs/turbo/coe-mavens/zxuechen/LightGaussian/result', 'load_path', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    # render_path_spiral
    # render_path_spherical
    for idx, pose in enumerate(tqdm(generate_xyz_path(views), desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    out_path = os.path.join(render_path,'video')
    makedirs(out_path, exist_ok=True)
    #fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
    # fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    first_image_path = os.path.join(render_path, '{0:05d}'.format(0) + ".png")
    first_image = Image.open(first_image_path)
    size = (first_image.width, first_image.height)
    fps = float(72)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)
    if not writer.isOpened():
        print("Error: Failed to open video writer.")
    else:
        print('Video writer opened successfully.')
        image_dir = render_path
        for file_name in sorted(os.listdir(render_path)):
            if file_name.endswith('.png'):
                image_path = os.path.join(image_dir, file_name)
                rgb = np.array(Image.open(image_path))
                rgb = rgb.astype('uint8')
                writer.write(rgb[:,:,::-1])
        writer.release()


def render_rt_trace(trace_path, iteration, views, gaussians, pipeline, background,save_video):
    render_path = os.path.join("/home/zxuechen/LightGaussian/OUTPUT", 'video_trace', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    for idx, pose in enumerate(tqdm(generate_rt_path(views,trace_path), desc="Rendering progress")):
        pose = np.array(pose)
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    out_path = os.path.join(render_path,'video')
    makedirs(out_path, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    first_image_path = os.path.join(render_path, '{0:05d}'.format(0) + ".png")
    first_image = Image.open(first_image_path)
    size = (first_image.width, first_image.height)
    fps = float(72)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)
    if not writer.isOpened():
        print("Error: Failed to open video writer.")
    else:
        print('Video writer opened successfully.')
        image_dir = render_path
        for file_name in sorted(os.listdir(render_path)):
            if file_name.endswith('.png'):
                image_path = os.path.join(image_dir, file_name)
                rgb = np.array(Image.open(image_path))
                rgb = rgb.astype('uint8')
                writer.write(rgb[:,:,::-1])
        writer.release()

def gaussian_render(model_path, iteration, views, gaussians, pipeline, background, args):
    views = views[:10] #take the first 10 views and check gaussian view point 
    render_path = os.path.join(model_path, 'video', "gaussians_{}_std{}".format(iteration, args.std))
    makedirs(render_path, exist_ok=True)

    for i, view in enumerate(views):
        rendering = render(view, gaussians, pipeline, background)["render"]
        sub_path = os.path.join(render_path,"view_"+str(i))
        makedirs(sub_path ,exist_ok=True)
        torchvision.utils.save_image(rendering, os.path.join(sub_path, "gt"+'{0:05d}'.format(i) + ".png"))
        for j in range(10):
            n_view = copy.deepcopy(view)
            g_view = gaussian_poses(n_view, args.mean, args.std)
            rendering = render(g_view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(sub_path, '{0:05d}'.format(j) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, video: bool, circular:bool, radius: float, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_vq= args.load_vq)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        # if circular:
        #     render_circular_video(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,radius)
        # # by default generate ellipse path, other options include spiral, circular, or other generate_xxx_path function from utils.pose_utils 
        # # Modify trajectory function in render_video's enumerate 
        # if video:
        #     render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        # #sample virtual view 
        # if args.gaussians:
        #     gaussian_render(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)
        if args.trace:
            if args.trace_type == 'real':
                render_xyz_trace(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,save_video=True)
            if args.trace_type == 'synthetic':
                render_rt_trace(args.trace_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,save_video=True)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--circular", action="store_true")
    parser.add_argument("--radius", default=5, type=float)
    parser.add_argument("--gaussians", action="store_true")
    parser.add_argument("--mean", default=0, type=float)
    parser.add_argument("--std", default=0.03, type=float)
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument("--trace_path", type=str)
    parser.add_argument("--trace_type", default='synthetic',type=str)


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    print("Trace path " + args.trace_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.video, args.circular, args.radius, args)