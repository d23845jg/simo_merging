import torch
import wandb

from models.dinov2.mtl.multitasker import *
from training.create_dataset import *
from training.create_network import *


def initialize_wandb(project, group, job_type, mode, config):
    # Initialize wandb
    wandb.init(
        project=project,
        group=group,
        job_type=job_type,  # "mtl", "task_specific", "model_merging"
        mode=mode,
        force=True,
        save_code=True,
        dir="logs/",
    )

    # track hyperparameters and run metadata
    wandb.config.update(config)

    if wandb.run is not None:
        INVALID_PATHS = ["models", "logs", "dataset"]
        wandb.run.log_code(
            exclude_fn=lambda path: any(
                [
                    path.startswith(os.path.expanduser(os.getcwd() + "/" + i))
                    for i in INVALID_PATHS
                ]
            )
        )
    return wandb


def get_data_loaders(config, model_merging=False):
    dataset_name = (
        config["model_merging"]["dataset"]
        if model_merging
        else config["training_params"]["dataset"]
    )

    # Check if dataset is in the config paths
    if dataset_name not in config["dataset_paths"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_path = config["dataset_paths"][dataset_name]

    # Initialize datasets based on the selected dataset
<<<<<<< Updated upstream
    if dataset_name == 'nyuv2':
        train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
        test_set = NYUv2(root=dataset_path, train=False)
    
    elif dataset_name == 'cityscapes':
        train_set = CityScapes(root=dataset_path, train=True, augmentation=True)
        test_set = CityScapes(root=dataset_path, train=False)
    
=======
    if dataset_name == "nyuv2":
        train_set = NYUv2(root=dataset_path, mode="train", augmentation=True)
        val_set = NYUv2(root=dataset_path, mode="val")
        val, test = random_split(val_set, [400, 254], torch.Generator().manual_seed(42))

        train_set = NYUv2(root=dataset_path, mode="train", augmentation=True)
        val_set = SplitNYUv2(
            root=dataset_path,
            list_of_indices=val.indices,
            mode="test",
            augmentation=True,
        )
        test_set = SplitNYUv2(
            root=dataset_path, list_of_indices=test.indices, mode="test"
        )

    elif dataset_name == "cityscapes":
        train_set = CityScapes(root=dataset_path, mode="train", augmentation=True)
        train, val = random_split(
            train_set, [2380, 595], torch.Generator().manual_seed(42)
        )

        train_set = SplitCityScapes(
            root=train.dataset.root,
            list_of_indices=train.indices,
            mode="train",
            augmentation=True,
        )
        val_set = SplitCityScapes(
            root=train.dataset.root,
            list_of_indices=val.indices,
            mode="val",
            augmentation=True,
        )
        test_set = CityScapes(root=dataset_path, mode="test")

    elif dataset_name == "taskonomy":
        train_set = Taskonomy(
            root=dataset_path,
            model_whitelist=[
                "allensville",
                "coffeen",
                "collierville",
                "leonardo",
                "merom",
                "pinesdale",
                "ranchester",
                "stockman",
            ],
            model_limit=2_000,
            label_set=[
                "depth_zbuffer",
                "normal",
                "segment_semantic",
                "keypoints2d",
                "edge_texture",
            ],
            output_size=(256, 256),
            augment=True,
        )
        val_set = Taskonomy(
            root=dataset_path,
            model_whitelist=["beechwood", "corozal", "klickitat", "shelbyville"],
            model_limit=1_000,
            label_set=[
                "depth_zbuffer",
                "normal",
                "segment_semantic",
                "keypoints2d",
                "edge_texture",
            ],
            output_size=(256, 256),
            augment=True,
        )
        test_set = Taskonomy(
            root=dataset_path,
            model_whitelist=["ihlen", "mcdade", "onaga", "tolstoy"],
            model_limit=1_000,
            label_set=[
                "depth_zbuffer",
                "normal",
                "segment_semantic",
                "keypoints2d",
                "edge_texture",
            ],
            output_size=(256, 256),
            augment=False,
        )

>>>>>>> Stashed changes
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for data loading")

    # Initialize data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config["training_params"]["batch_size"],
        shuffle=True,
<<<<<<< Updated upstream
        num_workers=4
=======
        num_workers=4,
        pin_memory=True,
        sampler=None,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=config["training_params"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
>>>>>>> Stashed changes
    )
    val_loader = None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config["training_params"]["batch_size"],
        shuffle=False,
<<<<<<< Updated upstream
        num_workers=4
=======
        num_workers=4,
        pin_memory=True,
        sampler=None,
>>>>>>> Stashed changes
    )

    return train_loader, val_loader, test_loader


"""
Define model saving and loading functions here.
"""


def torch_save(model, filename):
    metadata = {
        "model_class": model.__class__.__name__,
        "constructor_args": model.constructor_args,
        "head_tasks": model.head_tasks,
        "state_dict": model.state_dict(),
    }
    torch.save(metadata, filename)


def torch_load(filename):
    metadata = torch.load(filename, map_location="cpu")

    model_class_name = metadata["model_class"]
    model = globals()[model_class_name](**metadata["constructor_args"])

    model.load_state_dict(metadata["state_dict"])
    return model
