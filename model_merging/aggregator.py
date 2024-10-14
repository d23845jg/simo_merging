import torch
import wandb

from itertools import chain

from model_merging.task_vectors import MTLTaskVector
from model_merging.ties_utils import ties_merging
from model_merging.utils import state_dict_to_vector, vector_to_state_dict, topk_values_mask
# from model_merging.tallmask_utils import construct_consensus_mask, construct_tall_mask, load_tall_mask


def aggregate_task_vectors(task_vectors, mm_config):
    # Flattening out Checkpoints
    remove_keys = []
    flat_task_vectors = torch.vstack([state_dict_to_vector(task_vector.theta, remove_keys) for task_vector in task_vectors])

    # Aggregate Task Vectors
    merge_method = mm_config["model_merging"]["method"]
    merge_config = mm_config[merge_method]
    if merge_config["name"] == "ties":
        # TIES Merging
        merge_func = "dis-mean"
        merged_tv = ties_merging(flat_task_vectors, reset_thresh=merge_config["k"], merge_func=merge_func)
    elif merge_config["name"] in ["sum", "zeroshot", "average"]:
        # "sum" corresponds to Task Arithmetic (TA)
        # TA, zeroshot, weight average all construct the task vector with sum, but use different scaling factors.
        flat_task_vectors, _ = topk_values_mask(flat_task_vectors, K=merge_config["k"], return_mask=False)
        merged_tv = flat_task_vectors.sum(dim=0)
    # TODO:
    # elif merge_config["name"] == "tall_mask":
    #     # construct multi-task vector
    #     if merge_config["use_ties"]:
    #         print(f"Using TIES for constructing multi-task vector")
    #         merged_tv = ties_merging(flat_task_vectors, reset_thresh=20, merge_func=f"dis-sum")
    #     else:
    #         print(f"Using Task Arithmetic for constructing multi-task vector")
    #         flat_task_vectors, _ = topk_values_mask(flat_task_vectors, K=merge_config["k"], return_mask=False)
    #         merged_tv = flat_task_vectors.sum(dim=0)
    #     # get TALL masks
    #     if merge_config["load_masks"]:
    #         # load tall masks directly from storage
    #         eval_masks = load_tall_mask(remove_keys, ptm_check, config)
    #     else:
    #         print(f"=== Constructing TALL Mask ===")
    #         # construct tall masks
    #         eval_masks = construct_tall_mask(
    #             flat_task_vectors, flat_ft, flat_ptm, merged_tv, ptm_check, remove_keys, config
    #         )
    # elif merge_config["name"] == "consensus":  # consensus merging
    #     # construct consensus mask (assuming the TALL masks have already been constructed)
    #     consensus_mask = construct_consensus_mask(ptm_check, merge_config["prun_thre_k"], config, remove_keys)
    #     # construct multi-task vector
    #     if merge_config["use_ties"]:
    #         merged_tv = ties_merging(flat_task_vectors, reset_thresh=20, merge_func="dis-sum")
    #     else:
    #         flat_task_vectors, _ = topk_values_mask(
    #             flat_task_vectors, K=merge_config["k"], return_mask=False
    #         )  # top-k mag filtering
    #         merged_tv = flat_task_vectors.sum(dim=0)
    #     # apply the consensus mask to filter multi-task vector
    #     merged_tv = merged_tv * consensus_mask
    # elif merge_config["name"] == "mag_masking":
    #     # Magnitude masking baseline
    #     print(f"=== Using Magnitude Masking ===")
    #     merged_tv = flat_task_vectors.sum(dim=0)
    #     _, _, eval_masks = topk_values_mask(flat_task_vectors, K=merge_config["k"], return_mask=True)
    #     eval_masks = [vector_to_state_dict(mask, ptm_check, remove_keys=remove_keys) for mask in eval_masks]
    #     eval_masks = {key: value for key, value in zip(mm_config.DATASETS, eval_masks)}
    else:
        raise ValueError(f"Method {mm_config['model_merging']['name']} not defined.")

    shared_tv_state_dict = vector_to_state_dict(merged_tv, task_vectors[0].theta, remove_keys=remove_keys)
    task_specific_tv_state_dict = dict(chain(*(task_vector.tau.items() for task_vector in task_vectors)))
    mtl_task_vector = MTLTaskVector(theta=shared_tv_state_dict, tau=task_specific_tv_state_dict)
    mtl_task_vector.head_tasks = dict(chain(*(task_vector.head_tasks.items() for task_vector in task_vectors))) # Assume one ft model for each task (e.g. there can't be two seg tasks)
    print("Norm of shared task vector: ", mtl_task_vector.norm())

    # if merge_config["name"] not in ["tall_mask", "mag_masking"]:
    #     eval_masks = None

    # return task_vector, eval_masks
    return mtl_task_vector
