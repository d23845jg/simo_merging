# Single-Input Multiple-Output (SIMO) Model Merging

Paper: Single-Input Multiple-Output Model Merging: Leveraging Foundation Models for Multi-Task Learning — https://arxiv.org/abs/2504.11268

This repository provides a practical SIMO model-merging toolkit built on DINOv2 backbones and task-specific heads. It implements task-vector construction and multiple aggregation strategies to combine single-task checkpoints into a single encoder with multiple task decoders, with evaluation on NYUv2, Cityscapes, and Taskonomy.

Key ideas from the report:
- SIMO differs from standard single-output merging: task-specific decoders and diverse losses can misalign representations after merging.
- Simple fixes re-align encoder features with fixed heads, enabling effective multi-task behavior without costly joint training.

![System diagram](figs/system.png)


## Features

- Task-vector construction (pretrained vs. fine-tuned) with shared encoder and fixed task heads.
- Merging methods implemented in `model_merging/`:
	- sum: Task Arithmetic (TA)
	- average: equally-scaled TA
	- zeroshot: apply heads to the base encoder without merging (alpha=0)
	- ties: TIES-Merging with pruning (k)
	- tall_mask: Task-ALigned Layerwise masking and evaluation
	- consensus: consensus mask over TALL masks
- Evaluation utilities with normalized metrics against task-specific baselines and optional W&B logging.


## Repository structure

- `config/`: Jinja2-powered YAML templates for training and merging.
- `model_merging/`: Task vectors, aggregators, masks, and evaluation logic.
- `models/dinov2/`: Backbone, decoder heads, and layers adapted from DINOv2.
- `training/`: Datasets and utilities for metrics/visualization.
- `notebooks/`: End-to-end examples and analysis (SIMO merging, traditional merging, task relationships, eval playground).


## Datasets

Supported datasets (as in the paper):
- NYUv2 — tasks: seg, depth, normal
- Cityscapes — tasks: seg, part_seg, disp
- Taskonomy (subset) — tasks: segment_semantic, depth_zbuffer, normal, keypoints2d, edge_texture

Point `config.dataset_paths` to your local copies. Example:

```yaml
dataset_paths:
	nyuv2: /path/to/nyuv2
	cityscapes: /path/to/cityscapes
	taskonomy: /path/to/taskonomy
```


## Configuration
Templates live in `config/` and `config/model_merging/`. Start from `config/mtl.yaml.j2` and edit the following keys:

- `wandb`: project/mode
- `dataset_paths`: absolute paths for nyuv2/cityscapes/taskonomy
- `training_params`: dataset, batch size, etc. (used by evaluators and loaders)
- `model_merging`:
	- `method`: one of `sum`, `average`, `zeroshot`, `ties`, `tall_mask`, `consensus`
	- `ft_model_files`: list of task-specific checkpoint files to merge
	- `dataset`, `network`, `out_dir`
	- `num_scaling_coef_samples`: grid size for searching alpha (merging scale)
	- `specify_lambda`: fix alpha; set to `None` to grid search

The method-specific defaults included via Jinja are in `config/model_merging/*.yaml.j2`.


## End-to-end: merge and evaluate

Below is a minimal example that constructs task vectors from task-specific models, merges them, and evaluates the merged model. It assumes your task-specific checkpoints were saved using `utils.torch_save` so that constructor args and `head_tasks` are recoverable.

```python
import yaml
from jinja2 import Environment, FileSystemLoader
import torch

from model_merging.task_vectors import MTLTaskVector
from model_merging.aggregator import aggregate_task_vectors
from model_merging.eval_utils import perform_eval_with_merged_vector
from utils import torch_load

# 1) Load config
env = Environment(loader=FileSystemLoader("."))
config = yaml.safe_load(env.get_template("config/mtl.yaml.j2").render())

# 2) Build per-task vectors by diffing fine-tuned vs. pretrained model
#    - pretrained checkpoint can be a path saved with utils.torch_save, or an nn.Module
#    - for best results, use the same backbone/heads architecture across tasks
pt_checkpoint = "<path-to-base-pretrained-checkpoint>.pt"  # or a model object

task_vectors = {}
for ft_path in config["model_merging"]["ft_model_files"]:
		# The fine-tuned checkpoint should contain a single task head
		ft_model = torch_load(ft_path)
		tv = MTLTaskVector(pretrained_checkpoint=pt_checkpoint, finetuned_checkpoint=ft_model)
		# Use the (single) head name as the key, e.g., "seg", "depth"
		task_name = list(ft_model.head_tasks.keys())[0]
		task_vectors[task_name] = tv

# 3) Merge task vectors according to the chosen method
mtl_task_vector, eval_masks = aggregate_task_vectors(task_vectors, config)

# 4) Evaluate on val, pick best alpha/masks, then test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_results = perform_eval_with_merged_vector(pt_checkpoint, mtl_task_vector, config, eval_masks=eval_masks)
print(final_results["test"])  # metrics per task and aggregated
```

Tips:
- `zeroshot`: sets alpha=0.0 (do not add task vectors) but reuse decoders.
- `average`: uses alpha=1/num_tasks.
- `tall_mask`/`consensus`: produce per-task masks to re-align features after merging.
- Normalized metrics are reported against single-task baselines (see `model_merging/eval_utils.py`).


## Notebooks
Open and run the notebooks for paper-style reproductions and analysis:
- `notebooks/simo_merging.ipynb`: main SIMO merging flows
- `notebooks/traditional_model_merging.ipynb`: traditional merging baselines
- `notebooks/task_relationships.ipynb`: relationships across tasks via vectors
- `notebooks/eval_playground.ipynb`: ad-hoc evals


## Citation
```bibtex
@article{garcia2025single,
  title={Single-Input Multi-Output Model Merging: Leveraging Foundation Models for Dense Multi-Task Learning},
  author={Garcia Giraldo, Juan and Dimitriadis, Nikolaos and Wang, Ke and Frossard, Pascal},
  journal={arXiv preprint arXiv:2504.11268},
  year={2025}
}
```


## Acknowledgements
- MTL DINOv2 components under `models/dinov2/`, and are adapted from the official project: https://github.com/facebookresearch/dinov2
- This repo uses ideas from prior model-merging repos [TALL masks](https://github.com/nik-dim/tall_masks) and [Auto-Lambda](https://github.com/lorenmt/auto-lambda).
