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
import imageio
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)


def tensor_stats(tensor):
    """
    计算PyTorch张量的统计特征。

    参数:
    tensor (torch.Tensor): 输入张量。

    返回:
    dict: 包含张量统计特征的字典。
    """
    stats = {
        'max': tensor.max(),  # 最大值
        'min': tensor.min(),  # 最小值
        'mean': tensor.mean(),  # 均值
        'median': torch.median(tensor),  # 中位数
        'std': tensor.std(),  # 标准差
        'var': tensor.var(),  # 方差
        'sum': tensor.sum(),  # 总和
        'nonzero_count': torch.count_nonzero(tensor),  # 非零元素数量
    }
    return stats

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor, checkpoint):
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds_{scale_factor}")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{scale_factor}")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"test_preds")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt")
    video_path = os.path.join(model_path, name, "ours_{}".format(iteration), "output.mp4")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    print(f"gaussians.get_xyz.shape: {gaussians.get_xyz.shape}, checkpoint=='': {checkpoint == ''}")
    print(tensor_stats(gaussians.get_scaling_with_3D_filter))
    print(tensor_stats(gaussians.get_opacity_with_3D_filter))
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if checkpoint != "":
            pose = gaussians.get_RT(view.uid)
            # rendering = render(view, gaussians, pipeline, background,
            #                    kernel_size=kernel_size,
            #                    camera_pose=pose,
            #                    update_pose=True
            #                    )["render"]
            rendering = render(view, gaussians, pipeline, background,
                               kernel_size=kernel_size,
                               update_pose=False
                               )["render"]
        else:
            rendering = render(view, gaussians, pipeline, background,
                               kernel_size=kernel_size,
                               update_pose=False
                               )["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        print(f'save in {os.path.join(render_path, "{0:05d}".format(idx) + ".png")}')
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    images_to_video(render_path, video_path, fps=12)
    print(f"save in {render_path}")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, checkpoint):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if checkpoint != "":
            print(f"loading checkpoint in {checkpoint}")
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params)
            scene.gaussians = gaussians
        else:
            print(f"nonono loading checkpoint")
        scale_factor = dataset.resolution
        bg_color = [1, 1, 1] if dataset.white_background else [102 / 255, 102 / 255, 102 / 255]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor, checkpoint=checkpoint)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = "")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.start_checkpoint)