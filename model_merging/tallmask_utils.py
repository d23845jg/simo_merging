import copy
import os
from typing import List, Optional

import numpy as np
import torch
import wandb

from .utils import state_dict_to_vector, vector_to_state_dict


def generate_task_masks(
    flat_task_vectors: torch.Tensor,
    mtl_task_vector: torch.Tensor,
    tall_mask_lambda: float = 1.0,
) -> torch.Tensor:
    """
    Generate task-specific TALL masks
    TALL masks are generated as: mask_t = |theta_0 - theta_t| > |theta_mt - theta_t| * lambda

    Args:
        tv_flat_checks: individual task vectors
        flat_ft: individual theta_t (fine-tuned weights)
        flat_ptm: theta_0 (pre-trained weight)
        tv: multi-task vector
        tall_mask_lambda: hyper-parameter lambda for generating TALL masks
    Returns:
        final_mask: generated TALL masks with the given lambda, in shape (n_task, n_parameter)
    """

    print(f"Generating TALL masks.")

    # compare the l1 distance, scaled with hyper-parameter lambda
    mask = (
        flat_task_vectors.abs()
        > (mtl_task_vector - flat_task_vectors).abs() * tall_mask_lambda
    )

    print(
        f"Average sparsity for the mask with tall_mask_lambda of {tall_mask_lambda}: {mask.float().mean():.4f}"
    )
    # log_wandb_mask_sparsity(mask)
    return mask


def construct_tall_mask(
    task_vectors,
    flat_task_vectors: torch.Tensor,
    merged_tv: torch.Tensor,
    remove_keys: List[str],
):
    """
    Construct TALL masks for all tasks for each lambda, and store in dictionary

    Args:
        task_vectors: individual task vectors
        flat_task_vectors: individual task vectors
        merged_tv: multi-task vector
        ptm_check: pre-trained weight as state dictionary
        remove_keys: the keys to be removed when converting between dictionary and vector
    Returns:
        tall_masks: constructed TALL masks in dictionary format of {lambda: {task: mask}}
    """
    tall_masks = {}
    for tall_mask_lambda in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        # generate tall masks for each lambda
        masks_at_scale = generate_task_masks(
            flat_task_vectors, merged_tv, tall_mask_lambda=tall_mask_lambda
        )
        # convert vectors to dictionary
        masks_at_scale = [
            vector_to_state_dict(
                mask, list(task_vectors.values())[0].theta, remove_keys=remove_keys
            )
            for mask in masks_at_scale
        ]
        # store the masks with {task: mask}
        tall_masks[tall_mask_lambda] = {
            key: value for key, value in zip(task_vectors.keys(), masks_at_scale)
        }
    return tall_masks


def find_optimal_mask(val_metrics, eval_masks, config, save_masks=True):
    """
    Respectively finds the optimal mask for each data task based on the validation accuracy

    Args:
        val_metrics: validation metrics for each lambda
        eval_masks: all generated masks

    Returns:
        best_masks_for_test: the best masks for each task, selected based on validation accuracy from each task
        best_val_metrics: best validation metrics for each task
    """
    print("=" * 39, "finding optimal lambdas for masks", "=" * 39)

    # transpose the dict from lambda-task to task-lambda
    transposed_dict = {}
    for key, inner_dict in val_metrics.items():
        for inner_key, value in inner_dict.items():
            if inner_key not in transposed_dict:
                transposed_dict[inner_key] = {}
            transposed_dict[inner_key][key] = value

    # for each task, find the best lambda
    max_subkeys = {
        task: (
            max(lambdas, key=lambda k: lambdas[k].get("metric", float("-inf")))
            if task in ["seg", "part_seg", "segment_semantic"] or "class" in task
            else min(
                lambdas, key=lambda k: lambdas[k].get("metric", float("inf"))
            )  # if task in ["depth", "normal", "disp"]:
        )
        for task, lambdas in transposed_dict.items()
        if any(isinstance(val, dict) and "metric" in val for val in lambdas.values())
    }

    # select the best mask for each task, which will be used for testing later
    best_masks_for_test = {}
    best_val_metrics = {}
    # respectively for each task:
    for task in max_subkeys:
        # select the lambda which achieves the best valdiation accuracy
        best_lambda = float(max_subkeys[task])
        print(f"Best lambda for {task} is {best_lambda}")
        # select the mask based on the selected lambda, save as dictionaries
        best_masks_for_test[task] = eval_masks[best_lambda][task]
        # save the best validation metric based on the selected lambda
        best_val_metrics[task] = val_metrics[best_lambda][task]

    # save the best masks in disk
    if save_masks and not config["tall_mask"]["load_masks"]:
        mask_name = (
            f"TALL_mask_{config['model_merging']['dataset']}.npy"
            if not config["tall_mask"]["use_ties"]
            else f"TALL_mask_ties_{config['tall_mask']['ties_agg']}_{config['model_merging']['dataset']}.npy"
        )
        torch.save(
            best_masks_for_test,
            os.path.join(config["model_merging"]["out_dir"], mask_name),
        )

    return best_masks_for_test, best_val_metrics


def load_tall_mask(remove_keys, config):
    """Loads TALL masks from disk, unpack and transform to state dictionaries."""
    try:
        if config["tall_mask"]["use_ties"]:
            print("==== Loading TALL Masks built with TIES ====")
            tall_masks = torch.load(
                os.path.join(
                    config["model_merging"]["out_dir"],
                    f"TALL_mask_ties_{config['tall_mask']['ties_agg']}_{config['model_merging']['dataset']}.npy",
                )
            )
        else:
            print("==== Loading TALL Masks built with Task Arithmetic ====")
            tall_masks = torch.load(
                os.path.join(
                    config["model_merging"]["out_dir"],
                    f"TALL_mask_{config['model_merging']['dataset']}.npy",
                )
            )
    except:
        raise Exception("TALL Masks are not constructed yet.")
    return tall_masks


def construct_consensus_mask(prun_thre_k, config, remove_keys=[]):
    """
    Generate consensus mask by filtering out least-used parameters

    Args:
        prun_thre_k: weight-pruning threshold, stands for the least number of activated tasks for a parameter to be preserved from pruning
                if prun_thre_k is set to 2: remove both catastrophic and selfish weights;
                if prun_thre_k is set to 1: remove only catastrophic weights;
                if prun_thre_k is set to 0: remove no weights -> reduce to TA or TIES
                if prun_thre_k is set to > num_tasks: remove all weights -> reduce to zero-shot
    Returns:
        consensus_mask_vector: constructed consensus mask as vector (boolean in shape (n_parameter, ))
    """

    print("==== Generating Consensus Mask ====")
    # load TALL masks (in shape (n_task, n_parameter))
    tall_masks = load_tall_mask(remove_keys, config)
    tall_masks = list(tall_masks.values())

    # generate consensus masks
    consensus_mask = copy.deepcopy(tall_masks[0])
    for key, value in consensus_mask.items():
        consensus_mask[key] = torch.zeros_like(value)
        # count for each parameter, the tasks it has been activated for
        for mask in tall_masks:
            consensus_mask[key] = consensus_mask[key] + mask[key].float()
        # filter out the least-activated parameters based on given threshold
        consensus_mask[key] = consensus_mask[key].float() >= prun_thre_k
    consensus_mask_vector = state_dict_to_vector(
        consensus_mask, remove_keys=remove_keys
    )

    return consensus_mask_vector
