import os
# import cv2
import random
import torch
import fnmatch

import numpy as np
# import panoptic_parts as pp
import torch.utils.data as data
import matplotlib.pylab as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from PIL import Image
from torchvision.datasets import CIFAR100



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


class NYUv2(data.Dataset):
    """
    NYUv2 dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Semantic Segmentation,
        2. Depth prediction,
        3. Surface Normal prediction,
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]
    """
<<<<<<< Updated upstream
    def __init__(self, root, train=True, augmentation=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))
        self.noise = torch.rand(self.data_len, 1, 224, 224)

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0)).float()
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index))).long()
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0)).float()
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0)).float()
        noise = self.noise[index].float()

        data_dict = {'im': image, 'seg': semantic, 'depth': depth, 'normal': normal, 'noise': noise}

        # apply data augmentation if required
        if self.augmentation:
            # data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)
            data_dict = DataTransform(crop_size=[224, 224], scales=[1.0, 1.2, 1.5])(data_dict)

        im = 2. * data_dict.pop('im') - 1.  # normalised to [-1, 1]
=======

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
>>>>>>> Stashed changes
        return im, data_dict

    def __len__(self):
        return self.data_len


<<<<<<< Updated upstream
class CityScapes(data.Dataset):
=======
class SplitNYUv2(Dataset):
>>>>>>> Stashed changes
    """
    CityScapes dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Semantic Segmentation,
        2. Part Segmentation,
        3. Disparity Estimation (Inverse Depth),
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]
    """
<<<<<<< Updated upstream
    def __init__(self, root, train=True, augmentation=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.png'))
        self.noise = torch.rand(self.data_len, 1, 256, 256) if self.train else torch.rand(self.data_len, 1, 256, 512)

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(plt.imread(self.data_path + '/image/{:d}.png'.format(index)), -1, 0)).float()
        disparity = cv2.imread(self.data_path + '/depth/{:d}.png'.format(index), cv2.IMREAD_UNCHANGED).astype(np.float32)
        disparity = torch.from_numpy(self.map_disparity(disparity)).unsqueeze(0).float()
        seg = np.array(Image.open(self.data_path + '/seg/{:d}.png'.format(index)), dtype=float)
        seg = torch.from_numpy(self.map_seg_label(seg)).long()
        part_seg = np.array(Image.open(self.data_path + '/part_seg/{:d}.tif'.format(index)))
        part_seg = torch.from_numpy(self.map_part_seg_label(part_seg)).long()
        noise = self.noise[index].float()

        data_dict = {'im': image, 'seg': seg, 'part_seg': part_seg, 'disp': disparity, 'noise': noise}

        # apply data augmentation if required
        if self.augmentation:
            data_dict = DataTransform(crop_size=[256, 256], scales=[1.0])(data_dict)

        im = 2. * data_dict.pop('im') - 1.  # normalised to [-1, 1]
=======

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
>>>>>>> Stashed changes
        return im, data_dict

    def map_seg_label(self, mask):
        # source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        mask_map = np.zeros_like(mask)
        mask_map[np.isin(mask, [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30])] = -1
        mask_map[np.isin(mask, [7])] = 0
        mask_map[np.isin(mask, [8])] = 1
        mask_map[np.isin(mask, [11])] = 2
        mask_map[np.isin(mask, [12])] = 3
        mask_map[np.isin(mask, [13])] = 4
        mask_map[np.isin(mask, [17])] = 5
        mask_map[np.isin(mask, [19])] = 6
        mask_map[np.isin(mask, [20])] = 7
        mask_map[np.isin(mask, [21])] = 8
        mask_map[np.isin(mask, [22])] = 9
        mask_map[np.isin(mask, [23])] = 10
        mask_map[np.isin(mask, [24])] = 11
        mask_map[np.isin(mask, [25])] = 12
        mask_map[np.isin(mask, [26])] = 13
        mask_map[np.isin(mask, [27])] = 14
        mask_map[np.isin(mask, [28])] = 15
        mask_map[np.isin(mask, [31])] = 16
        mask_map[np.isin(mask, [32])] = 17
        mask_map[np.isin(mask, [33])] = 18
        return mask_map

    def map_part_seg_label(self, mask):
        # https://panoptic-parts.readthedocs.io/en/stable/api_and_code.html
        # https://arxiv.org/abs/2004.07944
        mask = pp.decode_uids(mask, return_sids_pids=True)[-1]
        mask_map = np.zeros_like(mask)  # background
        mask_map[np.isin(mask, [2401, 2501])] = 1    # human/rider torso
        mask_map[np.isin(mask, [2402, 2502])] = 2    # human/rider head
        mask_map[np.isin(mask, [2403, 2503])] = 3    # human/rider arms
        mask_map[np.isin(mask, [2404, 2504])] = 4    # human/rider legs
        mask_map[np.isin(mask, [2601, 2701, 2801])] = 5  # car/truck/bus windows
        mask_map[np.isin(mask, [2602, 2702, 2802])] = 6  # car/truck/bus wheels
        mask_map[np.isin(mask, [2603, 2703, 2803])] = 7  # car/truck/bus lights
        mask_map[np.isin(mask, [2604, 2704, 2804])] = 8  # car/truck/bus license_plate
        mask_map[np.isin(mask, [2605, 2705, 2805])] = 9  # car/truck/bus chassis
        return mask_map

    def map_disparity(self, disparity):
        # https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
        # remap invalid points to -1 (not to conflict with 0, infinite depth, such as sky)
        disparity[disparity == 0] = -1
        # reduce by a factor of 4 based on the rescaled resolution
        disparity[disparity > -1] = (disparity[disparity > -1] - 1) / (256 * 4)
        return disparity

    def __len__(self):
        return self.data_len


class CIFAR100MTL(CIFAR100):
    """
    CIFAR100 dataset, 20 tasks (grouped by coarse labels)
    Each task is a 5-label classification, with 2500 training and 500 testing number of data for each task.
    Modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, subset_id=0, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100MTL, self).__init__(root, train, transform, target_transform, download)
        # define coarse label maps
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

<<<<<<< Updated upstream
        self.coarse_targets = coarse_labels[self.targets]

        # filter the data and targets for the desired subset
        self.data = self.data[self.coarse_targets == subset_id]
        self.targets = np.array(self.targets)[self.coarse_targets == subset_id]

        # remap fine labels into 5-class classification
        self.targets = np.unique(self.targets, return_inverse=True)[1]

        # update semantic classes
        self.class_dict = {
            "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
            "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
            "food containers": ["bottle", "bowl", "can", "cup", "plate"],
            "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
            "household electrical device": ["clock", "computer_keyboard", "lamp", "telephone", "television"],
            "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
            "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
            "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
            "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
            "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
            "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
            "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
            "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
            "people": ["baby", "boy", "girl", "man", "woman"],
            "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
            "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
            "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
            "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
        }

        self.subset_class = list(self.class_dict.keys())[subset_id]
        self.classes = self.class_dict[self.subset_class]

=======
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
>>>>>>> Stashed changes
