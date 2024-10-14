import copy
import torch
from collections import OrderedDict


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]

    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector([value.reshape(-1) for key, value in sorted_shared_state_dict.items()])


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict["transformer.shared.weight"]
    return sorted_reference_dict


def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False
    return True


def topk_values_mask(M, K=0.7, return_mask=False, reshape_mask=False):
    if K == 100:
        # print("Not applying mask")
        if return_mask:
            return M, torch.ones_like(M), None
        else:
            return M, torch.ones_like(M)

    if K >= 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if reshape_mask:
        final_mask = final_mask.reshape(M.shape)

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    else:
        return M * final_mask, final_mask.float().mean(dim=1)