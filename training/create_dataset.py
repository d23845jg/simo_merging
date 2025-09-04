import fnmatch
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from PIL import Image, ImageOps
from torch.utils.data import Dataset, random_split


class DataTransform(object):
    def __init__(self, scales, crop_size, is_disparity=False):
        self.scales = scales
        self.crop_size = crop_size
        self.is_disparity = is_disparity

    def __call__(self, data_dict):
        if type(self.scales) == tuple:
            # Continuous range of scales
            sc = np.random.uniform(*self.scales)

        elif type(self.scales) == list:
            # Fixed range of scales
            sc = random.sample(self.scales, 1)[0]

        raw_h, raw_w = data_dict["im"].shape[-2:]
        resized_size = [int(raw_h * sc), int(raw_w * sc)]
        i, j, h, w = 0, 0, 0, 0  # initialise cropping coordinates
        flip_prop = random.random()

        for task in data_dict:
            if (
                len(data_dict[task].shape) == 2
            ):  # make sure single-channel labels are in the same size [H, W, 1]
                data_dict[task] = data_dict[task].unsqueeze(0)

            # Resize based on randomly sampled scale
            if task in ["im", "noise"]:
                data_dict[task] = transforms_f.resize(
                    data_dict[task], resized_size, Image.BILINEAR
                )
            elif task in ["normal", "depth", "seg", "part_seg", "disp"]:
                data_dict[task] = transforms_f.resize(
                    data_dict[task], resized_size, Image.NEAREST
                )

            # Add padding if crop size is smaller than the resized size
            if (
                self.crop_size[0] > resized_size[0]
                or self.crop_size[1] > resized_size[1]
            ):
                right_pad, bottom_pad = max(
                    self.crop_size[1] - resized_size[1], 0
                ), max(self.crop_size[0] - resized_size[0], 0)
                if task in ["im"]:
                    data_dict[task] = transforms_f.pad(
                        data_dict[task],
                        padding=(0, 0, right_pad, bottom_pad),
                        padding_mode="reflect",
                    )
                elif task in ["seg", "part_seg", "disp"]:
                    data_dict[task] = transforms_f.pad(
                        data_dict[task],
                        padding=(0, 0, right_pad, bottom_pad),
                        fill=-1,
                        padding_mode="constant",
                    )  # -1 will be ignored in loss
                elif task in ["normal", "depth", "noise"]:
                    data_dict[task] = transforms_f.pad(
                        data_dict[task],
                        padding=(0, 0, right_pad, bottom_pad),
                        fill=0,
                        padding_mode="constant",
                    )  # 0 will be ignored in loss

            # Random Cropping
            if i + j + h + w == 0:  # only run once
                i, j, h, w = transforms.RandomCrop.get_params(
                    data_dict[task], output_size=self.crop_size
                )
            data_dict[task] = transforms_f.crop(data_dict[task], i, j, h, w)

            # Random Flip
            if flip_prop > 0.5:
                data_dict[task] = torch.flip(data_dict[task], dims=[2])
                if task == "normal":
                    data_dict[task][0, :, :] = -data_dict[task][0, :, :]

            # Final Check:
            if task == "depth":
                data_dict[task] = data_dict[task] / sc

            if task == "disp":  # disparity is inverse depth
                data_dict[task] = data_dict[task] * sc

            if task in ["seg", "part_seg"]:
                data_dict[task] = data_dict[task].squeeze(0)
        return data_dict


class NYUv2(Dataset):
    """
    NYUv2 dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Semantic Segmentation,
        2. Depth prediction,
        3. Surface Normal prediction,
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]
    """

    def __init__(self, root, mode="train", augmentation=False):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        if mode == "train":
            self.data_path = os.path.join(self.root, "train")
        else:
            self.data_path = os.path.join(self.root, "val")

        self.data_len = len(
            fnmatch.filter(os.listdir(os.path.join(self.data_path, "image")), "*.npy")
        )

    def _load_npy(self, folder, index):
        return np.load(os.path.join(self.data_path, folder, f"{index}.npy"))

    def __getitem__(self, index):
        # Load data
        image = torch.from_numpy(
            np.moveaxis(self._load_npy("image", index), -1, 0)
        ).float()
        semantic = torch.from_numpy(self._load_npy("label", index)).long()
        depth = torch.from_numpy(
            np.moveaxis(self._load_npy("depth", index), -1, 0)
        ).float()
        normal = torch.from_numpy(
            np.moveaxis(self._load_npy("normal", index), -1, 0)
        ).float()

        data_dict = {"im": image, "seg": semantic, "depth": depth, "normal": normal}

        if self.augmentation:
            # data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)
            data_dict = DataTransform(crop_size=[280, 378], scales=[1.0, 1.2, 1.5])(
                data_dict
            )

        im = 2.0 * data_dict.pop("im") - 1.0  # Normalize to [-1, 1]
        return im, data_dict

    def __len__(self):
        return self.data_len


class SplitNYUv2(Dataset):
    """
    NYUv2 dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Semantic Segmentation,
        2. Depth prediction,
        3. Surface Normal prediction,
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]
    """

    def __init__(self, root, list_of_indices, mode="train", augmentation=False):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.list_of_indices = list_of_indices
        self.augmentation = augmentation

        if mode == "train":
            self.data_path = os.path.join(self.root, "train")
        else:
            self.data_path = os.path.join(self.root, "val")

        self.data_len = len(
            fnmatch.filter(os.listdir(os.path.join(self.data_path, "image")), "*.npy")
        )

    def _load_npy(self, folder, index):
        return np.load(os.path.join(self.data_path, folder, f"{index}.npy"))

    def __getitem__(self, index):
        index = self.list_of_indices[index]

        # Load data
        image = torch.from_numpy(
            np.moveaxis(self._load_npy("image", index), -1, 0)
        ).float()
        semantic = torch.from_numpy(self._load_npy("label", index)).long()
        depth = torch.from_numpy(
            np.moveaxis(self._load_npy("depth", index), -1, 0)
        ).float()
        normal = torch.from_numpy(
            np.moveaxis(self._load_npy("normal", index), -1, 0)
        ).float()

        data_dict = {"im": image, "seg": semantic, "depth": depth, "normal": normal}

        if self.augmentation:
            # data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)
            data_dict = DataTransform(crop_size=[280, 378], scales=[1.0, 1.2, 1.5])(
                data_dict
            )

        im = 2.0 * data_dict.pop("im") - 1.0  # Normalize to [-1, 1]
        return im, data_dict

    def __len__(self):
        return len(self.list_of_indices)


class CityScapes(Dataset):
    """
    We could further improve the performance with the data augmentation of NYUv2 defined in:
        [1] PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing
        [2] Pattern affinitive propagation across depth, surface normal and semantic segmentation
        [3] Mti-net: Multiscale task interaction networks for multi-task learning

        1. Random scale in a selected raio 1.0, 1.2, and 1.5.
        2. Random horizontal flip.

    Please note that: all baselines and MTAN did NOT apply data augmentation in the original paper.
    """

    def __init__(self, root, mode="train", augmentation=False):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        if mode == "test":
            self.data_path = os.path.join(self.root, "val")
        else:
            self.data_path = os.path.join(self.root, "train")

        self.data_len = len(
            fnmatch.filter(os.listdir(os.path.join(self.data_path, "image")), "*.npy")
        )

    def _load_npy(self, folder, index):
        return np.load(os.path.join(self.data_path, folder, f"{index}.npy"))

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(
            np.moveaxis(self._load_npy("image", index), -1, 0)
        ).float()
        semantic = torch.from_numpy(self._load_npy("label_19", index)).long()
        part_seg = torch.from_numpy(self._load_npy("part_seg", index)).long()
        disparity = torch.from_numpy(
            np.moveaxis(self._load_npy("depth", index), -1, 0)
        ).float()

        data_dict = {
            "im": image,
            "seg": semantic,
            "part_seg": part_seg,
            "disp": disparity,
        }

        # apply data augmentation if required
        if self.augmentation:
            data_dict = DataTransform(crop_size=[128, 256], scales=[1.0, 1.2, 1.5])(
                data_dict
            )

        im = 2.0 * data_dict.pop("im") - 1.0  # Normalize to [-1, 1]
        return im, data_dict

    def __len__(self):
        return self.data_len


class SplitCityScapes(Dataset):
    """
    We could further improve the performance with the data augmentation of NYUv2 defined in:
        [1] PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing
        [2] Pattern affinitive propagation across depth, surface normal and semantic segmentation
        [3] Mti-net: Multiscale task interaction networks for multi-task learning

        1. Random scale in a selected raio 1.0, 1.2, and 1.5.
        2. Random horizontal flip.

    Please note that: all baselines and MTAN did NOT apply data augmentation in the original paper.
    """

    def __init__(self, root, list_of_indices, mode="train", augmentation=False):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.list_of_indices = list_of_indices
        self.augmentation = augmentation

        if mode == "test":
            self.data_path = os.path.join(self.root, "val")
        else:
            self.data_path = os.path.join(self.root, "train")

        self.data_len = len(
            fnmatch.filter(os.listdir(os.path.join(self.data_path, "image")), "*.npy")
        )

    def _load_npy(self, folder, index):
        return np.load(os.path.join(self.data_path, folder, f"{index}.npy"))

    def __getitem__(self, index):
        index = self.list_of_indices[index]
        # load data from the pre-processed npy files
        image = torch.from_numpy(
            np.moveaxis(self._load_npy("image", index), -1, 0)
        ).float()
        semantic = torch.from_numpy(self._load_npy("label_19", index)).long()
        part_seg = torch.from_numpy(self._load_npy("part_seg", index)).long()
        disparity = torch.from_numpy(
            np.moveaxis(self._load_npy("depth", index), -1, 0)
        ).float()

        data_dict = {
            "im": image,
            "seg": semantic,
            "part_seg": part_seg,
            "disp": disparity,
        }

        # apply data augmentation if required
        if self.augmentation:
            data_dict = DataTransform(crop_size=[128, 256], scales=[1.0, 1.2, 1.5])(
                data_dict
            )

        im = 2.0 * data_dict.pop("im") - 1.0  # Normalize to [-1, 1]
        return im, data_dict

    def __len__(self):
        return len(self.list_of_indices)


# Adapted from: https://github.com/tstandley/taskgrouping/blob/bb1496e42ff442b7ac69e6f227060b8023325d07/taskonomy_loader.py
class Taskonomy(Dataset):
    def __init__(
        self,
        root,
        label_set=[
            "depth_zbuffer",
            "normal",
            "segment_semantic",
            "edge_occlusion",
            "reshading",
            "keypoints2d",
            "edge_texture",
            "principal_curvature",
            "rgb",
        ],
        model_whitelist=None,
        model_limit=None,
        output_size=None,
        return_filename=False,
        augment=False,
    ):
        self.root = root
        self.model_whitelist = model_whitelist
        self.model_limit = model_limit
        self.records = []

        for i, (where, subdirs, files) in enumerate(os.walk(os.path.join(root, "rgb"))):
            if subdirs != []:
                continue
            model = where.split("/")[-1]
            if self.model_whitelist is None or model in self.model_whitelist:
                full_paths = [os.path.join(where, f) for f in files]
                if isinstance(model_limit, tuple):
                    full_paths.sort()
                    full_paths = full_paths[model_limit[0] : model_limit[1]]
                elif model_limit is not None:
                    full_paths.sort()
                    full_paths = full_paths[:model_limit]
                self.records += full_paths

        self.label_set = label_set
        self.output_size = output_size
        self.return_filename = return_filename
        self.to_tensor = transforms.ToTensor()
        self.augment = augment

        if augment == "aggressive":
            print("Data augmentation is on (aggressive).")
        elif augment:
            print("Data augmentation is on (flip).")
        else:
            print("no data augmentation")

    def process_image(self, im):
        bands = im.getbands()
        if bands[0] == "L":  # Grayscale image
            im = np.array(im)
            im.setflags(write=1)
            im = torch.from_numpy(im).unsqueeze(0)
        else:  # RGB image
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                im = self.to_tensor(im)

        if self.output_size and self.output_size != im.size:
            im = transforms_f.resize(im, self.output_size, Image.BILINEAR)
        return im

    def rescale_image(
        self, im, new_scale=[-1.0, 1.0], current_scale=None, no_clip=False
    ):
        im = np.asarray(im, dtype=np.float32)
        if current_scale is not None:
            min_val, max_val = current_scale
            if not no_clip:
                im = np.clip(im, min_val, max_val)
            im = im - min_val
            im /= max_val - min_val
        min_val, max_val = new_scale
        im *= max_val - min_val
        im += min_val
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is an uint8 matrix of integers with the same width and height.
        """
        with torch.no_grad():
            file_name = self.records[index]

            flip_lr = self.augment and random.random() > 0.5
            flip_ud = self.augment == "aggressive" and random.random() > 0.5

            pil_im = Image.open(file_name)
            if flip_lr:
                pil_im = ImageOps.mirror(pil_im)
            if flip_ud:
                pil_im = ImageOps.flip(pil_im)

            im = self.process_image(pil_im)
            im = self.rescale_image(im, new_scale=[-1.0, 1.0], current_scale=[0.0, 1.0])

            # -----

            ys = {}
            mask = None
            if os.path.isfile(
                file_name.replace("rgb", "mask_valid", 1).replace(
                    "rgb", "depth_zbuffer", 1
                )
            ):
                self.label_set.append("mask")

            for i in self.label_set:
                if i == "mask" and mask is not None:
                    continue

                if i == "mask":
                    yfilename = file_name.replace("rgb", "mask_valid", 1).replace(
                        "rgb", "depth_zbuffer", 1
                    )
                elif i == "segment_semantic":
                    yfilename = file_name.replace("rgb", "segment_semantic", 1).replace(
                        "rgb", "segmentsemantic", 1
                    )
                else:
                    yfilename = file_name.replace("rgb", i)

                yim = Image.open(yfilename)

                if flip_lr:
                    yim = ImageOps.mirror(yim)
                if flip_ud:
                    yim = ImageOps.flip(yim)

                yim = self.process_image(yim)

                if "depth" in i:
                    yim = torch.log1p(yim) / torch.log(torch.tensor(2.0) ** 16)
                elif i == "keypoints2d":
                    yim = self.rescale_image(
                        yim.float(), new_scale=[-1.0, 1.0], current_scale=[0.0, 3129.0]
                    )
                elif i == "edge_texture":
                    yim = self.rescale_image(
                        yim.float(), new_scale=[-1.0, 1.0], current_scale=[0.0, 11041.0]
                    )
                elif i == "normal":
                    yim = self.rescale_image(
                        yim.float(), new_scale=[-1.0, 1.0], current_scale=[0.0, 1.0]
                    )
                    if flip_lr:
                        yim[0] *= -1.0
                    if flip_ud:
                        yim[1] *= -1.0
                elif i == "segment_semantic":
                    yim = yim.squeeze(0).long()
                elif i == "mask":
                    mask = yim.bool()
                    yim = mask

                ys[i] = yim

            if not "rgb" in self.label_set:
                ys["rgb"] = im

            if self.return_filename:
                return im, ys, file_name
            else:
                return im, ys

    def __len__(self):
        return len(self.records)
