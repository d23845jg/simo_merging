import torch
import torch.optim as optim
import os
import yaml
import wandb

from jinja2 import Environment, FileSystemLoader

from training.create_dataset import *
from training.create_network import *
from models.dinov2.mtl.multitasker import MTLDinoV2
from training.utils import create_task_flags, TaskMetric, eval
from utils import torch_save, get_data_loaders, initialize_wandb

# Login to wandb
import os
os.environ["WANDB_API_KEY"] = "e31760df895b69a4dbca617faa701601c012611b"
wandb.login(key="e31760df895b69a4dbca617faa701601c012611b")


# Options for training
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('config/mtl.yaml.j2')
rendered_yaml = template.render()
config = yaml.safe_load(rendered_yaml)


# Tune hyperparameters with sweeps
sweep_config = {
  "method": "bayes",
  "metric": {
    "name": f"backbone_head/test/metric/{config['training_params']['task']}",
    "goal": "minimize"
  },
  "parameters": {
    "lr_backbone_head": {
      "distribution": "uniform",
      "min": 0.00000001,
      "max": 0.0001,
    },
    "dino_output_layer": {
      "values": [
        [2, 5, 8, 11],
        [5, 7, 9, 11],
        [8, 9, 10, 11],
        [11],
      ]
    },
    "cls_token": {
      "values": [True, False]
    },
  },
  "early_terminate": {
    "type": "hyperband",
    "min_iter": 6,
  }
}

sweep_id = wandb.sweep(
  sweep_config, 
  project=config["wandb"]["project"], 
)



torch.manual_seed(config["training_params"]["seed"])
np.random.seed(config["training_params"]["seed"])
random.seed(config["training_params"]["seed"])

# device = torch.device(f"cuda:{config["training_params"]['gpu']}" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, scheduler, total_epochs, train_tasks, config, mode="head"):
  # Create data loaders
  train_loader, val_loader, test_loader = get_data_loaders(config)
  
  # Define metrics
  train_metric = TaskMetric(train_tasks, train_tasks, config["training_params"]["batch_size"], total_epochs, config["training_params"]["dataset"])
  test_metric = TaskMetric(train_tasks, train_tasks, config["training_params"]["batch_size"], total_epochs, config["training_params"]["dataset"])

  #  Training loop
  model.to(device)
  for epoch in range(total_epochs):
      model.train()
      train_dataset = iter(train_loader)
      for k in range(len(train_loader)):
          train_data, train_target = next(train_dataset)
          train_data = train_data.to(device)
          train_target = {task_id: train_target[task_id].to(device) for task_id in model.head_tasks}
          
          train_res = model(train_data, None, img_gt=train_target, return_loss=True)
          
          optimizer.zero_grad()
          train_res["total_loss"].backward()
          optimizer.step()
          scheduler.step()

          train_metric.update_metric(train_res, train_target)
      train_str = train_metric.compute_metric()
      
      wandb.log({
          **{f"{mode}/train/loss/{task_id}": train_res[task_id]["total_loss"] for task_id in model.head_tasks},
          **{f"{mode}/train/metric/{task_id}": train_metric.get_metric(task_id) for task_id in model.head_tasks}
      },) # step=epoch
      train_metric.reset()

      # evaluating
      test_str = eval(epoch, model, test_loader, test_metric, mode=mode)

      print(f"Epoch {epoch:04d} | TRAIN:{train_str} || TEST:{test_str} | Best: {config['training_params']['task'].title()} {test_metric.get_best_performance(config['training_params']['task']):.4f}")
  print("\n\n\n")
  
  
  
def hyper_param_search():
  # Initialize a new wandb run
  initialize_wandb(
    project=config["wandb"]["project"], 
    group=f"{config['training_params']['network']}", 
    job_type="task_specific", 
    mode=config["wandb"]["mode"], 
    config={
      "task": config['training_params']['task'],
      "network": config['training_params']['network'],
      "dataset": config['training_params']['dataset'],
      "epochs": config['training_params']['total_epochs'],
      "lr_head": config['training_params']['lr_head'],
      "batch_size": config['training_params']['batch_size'],
      "seed": config['training_params']['seed'],
    }
  )
    
  # config set by Sweep Controller
  wandb_config = wandb.config
  
  data_loader, _, _ = get_data_loaders(config)
  
  # Define tasks
  train_tasks = create_task_flags(config["training_params"]["task"], config["training_params"]["dataset"])
  print(f"Training Task: {config['training_params']['dataset'].title()} - {config['training_params']['task'].title()} in Single Task Learning Mode with {config['training_params']['network'].upper()}")

  # Initialize model
  model = MTLDinoV2(
    arch_name="vit_base",
    head_tasks=train_tasks,
    head_archs="linear_bins",
    out_index=wandb_config["dino_output_layer"],
    cls_token=wandb_config["cls_token"],
  )
  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Model: {config['training_params']['network'].title()} | Number of Trainable Parameters: {num_params/1e6:.2f}M \n\n")
  
  # Train Head
  model.freeze_shared_layers()
  optimizer = optim.AdamW(model.parameters(), lr=config["training_params"]["lr_head"], weight_decay=1e-4)
  scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["training_params"]["lr_head"], steps_per_epoch=len(data_loader), epochs=config["training_params"]["total_epochs"],  pct_start=0.1)
  print("Training Head")
  train(model, optimizer, scheduler, config["training_params"]["total_epochs"], train_tasks, config, mode="head")
  torch_save(model, f"{config['training_params']['out_dir']}/{config['training_params']['task']}/bins/{config['training_params']['task']}_head_model_lr_{wandb_config['lr_backbone_head']}_out_index_{wandb_config['dino_output_layer']}_cls_token_{wandb_config['cls_token']}.pt")
  
  # Train Backbone + Head
  model.freeze_shared_layers(requires_grad=True)
  optimizer = optim.AdamW(model.parameters(), lr=wandb_config["lr_backbone_head"], weight_decay=1e-4)
  scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=wandb_config["lr_backbone_head"], steps_per_epoch=len(data_loader), epochs=3 * config["training_params"]["total_epochs"],  pct_start=0.05)
  print("Training Backbone + Head")
  train(model, optimizer, scheduler, 3 * config["training_params"]["total_epochs"], train_tasks, config, mode="backbone_head")
  torch_save(model, f"{config['training_params']['out_dir']}/{config['training_params']['task']}/bins/{config['training_params']['task']}_model_lr_{wandb_config['lr_backbone_head']}_out_index_{wandb_config['dino_output_layer']}_cls_token_{wandb_config['cls_token']}.pt")


wandb.agent(sweep_id, hyper_param_search, count=10)
wandb.finish(quiet=True)