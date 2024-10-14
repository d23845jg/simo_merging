import copy
import numpy as np
import time
import torch
import wandb

from typing import Union, Dict, Any, Optional

from model_merging.task_vectors import MTLTaskVector
from training.utils import TaskMetric, eval
from utils import get_data_loaders

_Checkpoint = Union[str, torch.nn.Module]


def add_normalized_accuracy(metrics: Dict[str, Any], config: Dict[str, Any]):
    mtl = {
        'nyuv2': {'seg': 0.4337, 'depth': 0.5224, 'normal': 22.40},
        'cityscapes': {'seg': 0.5620, 'part_seg': 0.5274, 'disp': 0.84}
    }.get(config["model_merging"]["dataset"], {})

    delta_mtl = 0
    for task_id, value in metrics.items():
        if task_id in mtl:
            normalized_value = np.append(value[0], value[0][1] / mtl[task_id])
            metrics[task_id] = np.array(normalized_value)
        
        if task_id in ['seg', 'part_seg'] or 'class' in task_id:  # higher better
            delta_mtl += (metrics[task_id][1] - mtl[task_id]) / mtl[task_id]
        elif task_id in ['depth', 'normal', 'disp']:
            delta_mtl -= (metrics[task_id][1] - mtl[task_id]) / mtl[task_id]

    metrics['all'][0] = delta_mtl / len(mtl)
    return metrics


def evaluate_task_vector_at_coef(
    pt_checkpoint: _Checkpoint,
    task_vector: MTLTaskVector,
    config: Dict[str, Any],
    scaling_coef: float,
    use_val_dataset:bool=False,
    eval_masks=None,
):
    start_time = time.time()

    if eval_masks != None:
        assert config["model_merging"]["method"] in ["tall_mask", "mag_masking"]
    else:
        model = task_vector.apply_to(pt_checkpoint, scaling_coef=scaling_coef)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # if eval_masks != None:
    #     sparse_task_vector = copy.deepcopy(task_vector)
    #     # remove "Val" from dataset_name
    #     mask = eval_masks[dataset_name[:-3]] if "Val" in dataset_name else eval_masks[dataset_name]
    #     # apply mask to sparsify the task vectors with Hadamard product
    #     sparse_task_vector.vector = {k: sparse_task_vector.vector[k] * mask[k].bool().cpu() for k in mask.keys()}
    #     # reconstruct theta_t^
    #     model = sparse_task_vector.apply_to(pt_checkpoint, scaling_coef=1.0)

    _, val_loader, test_loader = get_data_loaders(config, model_merging=True)
    data_loader = val_loader if use_val_dataset else test_loader
    test_metric = TaskMetric(model.head_tasks, model.head_tasks, config["training_params"]["batch_size"], 1, config["model_merging"]["dataset"], include_mtl=True)
    eval(int(10 * scaling_coef), model, data_loader, test_metric) # 10 * scaling_coef is a hack to change it to an int
    coef_metrics = test_metric.metric

    coef_metrics = add_normalized_accuracy(coef_metrics, config)
    print(f"Total evaluation time: {time.time() - start_time:.2f}s")

    return coef_metrics


def evaluate_task_vector(
    pt_checkpoint: _Checkpoint,
    task_vector: MTLTaskVector,
    config: Dict[str, Any],
    use_val_dataset: bool=False,
    eval_masks=None
):
    info = {}

    if config["model_merging"]["method"] == "tall_mask" or eval_masks is not None:
        scaling_coef_range = [1.0]
    elif config["model_merging"]["method"] == "zeroshot":
        scaling_coef_range = [0.0]
    elif config["model_merging"]["method"] == "average":
        scaling_coef_range = [1 / len(task_vector.head_tasks)]
    elif config["model_merging"]["specify_lambda"] != "None":
        scaling_coef_range = [config["model_merging"]["specify_lambda"]]
    else:
        scaling_coef_range = np.linspace(0.0, 1.0, config["model_merging"]["num_scaling_coef_samples"])

    if config["model_merging"]["method"] == "tall_mask":
        if config["tall_mask"]["load_mask"]:
            print("=" * 43, f"Evaluating the loaded TALL masks", "=" * 43)
            info["loaded_mask"] = evaluate_task_vector_at_coef(
                pt_checkpoint, task_vector, config, 1.0, use_val_dataset, eval_masks,
            )
            print(f"Delta MTL: {round(info['loaded_mask']['all'][0], 2)}")
        else:
            for tall_mask_lambda in [0.2, 0.3, 0.4, 0.5, 0.6]:
                print("\n" * 2)
                print("=" * 43, f"tall_mask_lambda = {tall_mask_lambda:.2f}", "=" * 43)
                info[tall_mask_lambda] = evaluate_task_vector_at_coef(
                    pt_checkpoint, task_vector, config, 1.0, use_val_dataset, eval_masks[tall_mask_lambda],
                )
                print(f"Delta MTL: {round(info[tall_mask_lambda]['all'][0], 2)}")
    else:
        for scaling_coef in scaling_coef_range:
            print("\n" * 2)
            print("=" * 43, f"alpha = {scaling_coef:.2f}", "=" * 43)
            info[scaling_coef] = evaluate_task_vector_at_coef(
                pt_checkpoint, task_vector, config, scaling_coef, use_val_dataset, eval_masks
            )
            print(" | ".join([f"{task} metric: {round(info[scaling_coef][task][1], 4)}" for task in task_vector.head_tasks.keys()]))
            print(f"Delta MTL: {round(info[scaling_coef]['all'][0], 2)}")

    return info


def find_optimal_coef(
    results: Dict[str, Any],
    metric_name: str = "all",
    metric_index: int = 0,
    minimize: bool = False,
    control_metric: Optional[str] = None,
    control_metric_threshold: float = 0.0,
) -> float:
    """
    Finds the optimal coefficient based on the given results and metric.

    Args:
        results (Dict[str, Any]): A dictionary containing the results for different scaling coefficients.
        metric (str, optional): The metric to optimize. Defaults to "avg_normalized_top1".
        minimize (bool, optional): Whether to minimize the metric. Defaults to False.
        control_metric (str, optional): The control metric to check against. Defaults to None.
        control_metric_threshold (float, optional): The threshold value for the control metric. Defaults to 0.0.

    Returns:
        The optimal coefficient based on the given results and metric.
    """
    best_metric = float('inf') if minimize else float('-inf')
    best_coef = None

    for scaling_coef, metrics in results.items():
        # Check the control metric condition if applicable
        if control_metric and metrics.get(control_metric, float('inf')) < control_metric_threshold:
            continue

        current_metric = metrics.get(metric_name)[metric_index]
        if current_metric is None:
            continue  # Skip if the metric isn't found in the results

        # Update best_coef based on the metric comparison
        if (minimize and current_metric < best_metric) or (not minimize and current_metric > best_metric):
            best_metric = current_metric
            best_coef = scaling_coef
    return best_coef


def perform_eval_with_merged_vector(
    pt_checkpoint: _Checkpoint,
    task_vector: MTLTaskVector,
    config: Dict[str, Any], 
    eval_masks=None,
):
    assert task_vector is not None, "Task vector should not be None."
    if eval_masks is not None:
        assert config["model_merging"]["method"] in ["tall_mask", "mag_masking"]

    # evaluate on validation set
    # val_metrics = evaluate_task_vector(pt_checkpoint, task_vector, config, use_val_dataset=True, eval_masks=eval_masks)
    # TODO: remove this
    val_metrics = evaluate_task_vector(pt_checkpoint, task_vector, config, use_val_dataset=False, eval_masks=eval_masks)
    
    if config["model_merging"]["method"] == "tall_mask":
        if config["tall_mask"]["load_mask"]:
            best_masks_for_test = eval_masks
            best_val_metrics = val_metrics
        # else:
            # find the best mask individually for each task based on validation accuracy
            # best_masks_for_test, best_val_metrics = find_optimal_mask(val_metrics, eval_masks, args, save_masks=True)
    elif config["model_merging"]["method"] == "mag_masking":
        best_masks_for_test = eval_masks
        best_val_metrics = val_metrics[1.0]
    else:
        # find scaling factor alpha based on validation accuracy (for Task Arithmetic, TIES, Consensus Merging)
        optimal_coef = find_optimal_coef(val_metrics, metric_name="all", metric_index=0, minimize=False)
        best_val_metrics = val_metrics[optimal_coef]

    print("\n" * 2)

    # Evaluate on the test set with the optimal coefficients / masks
    if config["model_merging"]["method"] in ["tall_mask", "mag_masking"]:
        test_metrics = evaluate_task_vector_at_coef(
            pt_checkpoint, task_vector, config, 1.0, use_val_dataset=False, eval_masks=best_masks_for_test
        )
    else:
        test_metrics = evaluate_task_vector_at_coef(
            pt_checkpoint, task_vector, config, float(optimal_coef), use_val_dataset=False, eval_masks=None
        )

    print("=" * 100)
    for task_id, value in test_metrics.items():
        print(f"Test normalized metric for {task_id}: {value[-1]}")
    
    final_results = {"test": test_metrics, "val": val_metrics, "val_best": best_val_metrics}
    # log_results(final_results, args)

    return final_results