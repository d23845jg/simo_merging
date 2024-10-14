import torch
import wandb

from training.create_dataset import *
from training.create_network import *
from models.dinov2.mtl.multitasker import *


def initialize_wandb(project, group, job_type, mode, config):
    # Initialize wandb
    wandb.init(
        project=project,
        group=group,
        job_type=job_type, # "mtl", "task_specific", "model_merging"
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
            [path.startswith(os.path.expanduser(os.getcwd() + "/" + i)) for i in INVALID_PATHS]
            )
        )
    return wandb


def get_data_loaders(config, model_merging=False):
    dataset_name = config["model_merging"]["dataset"] if model_merging else config["training_params"]["dataset"]
    
    # Check if dataset is in the config paths
    if dataset_name not in config["dataset_paths"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataset_path = config["dataset_paths"][dataset_name]
    
    # Initialize datasets based on the selected dataset
    if dataset_name == 'nyuv2':
        train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
        test_set = NYUv2(root=dataset_path, train=False)
    
    elif dataset_name == 'cityscapes':
        train_set = CityScapes(root=dataset_path, train=True, augmentation=True)
        test_set = CityScapes(root=dataset_path, train=False)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for data loading")
    
    # Initialize data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config["training_params"]["batch_size"],
        shuffle=True,
        num_workers=4
    )
    val_loader = None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config["training_params"]["batch_size"],
        shuffle=False,
        num_workers=4
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
    model = globals()[model_class_name](**metadata['constructor_args'])
    
    model.load_state_dict(metadata["state_dict"])
    return model