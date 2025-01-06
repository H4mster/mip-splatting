import math
import os
import time
from pathlib import Path
from typing import List, NamedTuple, Tuple
import collections
import re

import numpy as np
import torch
import cv2
import PIL.Image
from PIL.ImageOps import exif_transpose
from plyfile import PlyData, PlyElement
import torchvision.transforms as tvf

from scene.colmap_loader import (
    qvec2rotmat, read_extrinsics_binary, rotmat2qvec,
)

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([
    tvf.ToTensor(),
    tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def save_time(time_dir, process_name, sub_time):
    if isinstance(time_dir, str):
        time_dir = Path(time_dir)
    time_dir.mkdir(parents=True, exist_ok=True)
    minutes, seconds = divmod(sub_time, 60)
    formatted_time = f"{int(minutes)} min {int(seconds)} sec"
    with open(time_dir / f'train_time.txt', 'a') as f:
        f.write(f'{process_name}: {formatted_time}\n')


def split_train_test(image_files, llffhold=8, n_views=None, verbose=True):
    test_idx = np.linspace(1, len(image_files) - 2, num=12, dtype=int)
    train_idx = [i for i in range(len(image_files)) if i not in test_idx]

    sparse_idx = np.linspace(0, len(train_idx) - 1, num=n_views, dtype=int)
    train_idx = [train_idx[i] for i in sparse_idx]

    if verbose:
        print(">> Spliting Train-Test Set: ")
        # print(" - sparse_idx:         ", sparse_idx)
        print(" - train_set_indices:  ", train_idx)
        print(" - test_set_indices:   ", test_idx)
    train_img_files = [image_files[i] for i in train_idx]
    test_img_files = [image_files[i] for i in test_idx]

    return train_img_files, test_img_files


def get_sorted_image_files(image_dir: str) -> Tuple[List[str], List[str]]:
    """
    Get sorted image files from the given directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - List of sorted image file paths
            - List of corresponding file suffixes
    """
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.JPG', '.PNG'}
    image_path = Path(image_dir)

    def extract_number(filename):
        match = re.search(r'\d+', filename.stem)
        return int(match.group()) if match else float('inf')

    image_files = [
        str(f) for f in image_path.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_extensions
    ]

    sorted_files = sorted(image_files, key=lambda x: extract_number(Path(x)))
    suffixes = [Path(file).suffix for file in sorted_files]

    return sorted_files, suffixes[0]



import collections

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def save_extrinsic(sparse_path, extrinsics_w2c, img_files, image_suffix):
    if isinstance(sparse_path, str):
        sparse_path = Path(sparse_path)
    images_txt_file = sparse_path / 'images.txt'
    images = {}

    for i, (w2c, img_file) in enumerate(zip(extrinsics_w2c, img_files)):
        name = Path(img_file).stem + image_suffix
        rotation_matrix = w2c[:3, :3]
        qvec = rotmat2qvec(rotation_matrix)
        tvec = w2c[:3, 3]

        images[i] = BaseImage(
            id=i,
            qvec=qvec,
            tvec=tvec,
            camera_id=i,
            name=name,
            xys=[],  # Empty list as we don't have 2D point information
            point3D_ids=[]  # Empty list as we don't have 3D point IDs
        )

    write_images_text(images, images_txt_file)
def save_intrinsics(sparse_path, focals, org_imgs_shape, imgs_shape):
    org_width, org_height = org_imgs_shape
    scale_factor_x = org_width / imgs_shape[2]
    scale_factor_y = org_height / imgs_shape[1]
    cameras_txt_file = sparse_path / 'cameras.txt'

    cameras = {}
    for i, focal in enumerate(focals):
        cameras[i] = Camera(
            id=i,
            model="PINHOLE",
            width=org_width,
            height=org_height,
            params=[focal*scale_factor_x, focal*scale_factor_y, org_width/2, org_height/2]
        )
    print(f' - scaling focal: ({focal}, {focal}) --> ({focal*scale_factor_x}, {focal*scale_factor_y})' )
    write_cameras_text(cameras, cameras_txt_file)

def write_images_text(images, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(
            (len(img.point3D_ids) for _, img in images.items())
        ) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [
                img.id,
                *img.qvec,
                *img.tvec,
                img.camera_id,
                img.name,
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")

def write_cameras_text(cameras, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")
