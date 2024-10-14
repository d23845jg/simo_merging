import torch

from typing import Union

from models.dinov2.mtl.multitasker import *
from training.create_network import *
from utils import torch_load

_Checkpoint = Union[str, nn.Module]


def symmetric_difference(A, B):
    """Returns the symmetric difference between two lists."""
    return list(set(A) ^ set(B))


class _TaskVector():
    def __init__(
        self,
        pretrained_state_dict=None,
        finetuned_state_dict=None,
        theta=None,
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if theta is not None:
            self.theta = theta
        else:
            assert pretrained_state_dict is not None and finetuned_state_dict is not None

            # the final task vector is the difference between finetuned and pretrained vectors.
            assert (pretrained_state_dict.keys() == finetuned_state_dict.keys()), f"State dicts have different keys: {symmetric_difference(pretrained_state_dict.keys(), finetuned_state_dict.keys())}."
            
            self.theta = {}
            for key in pretrained_state_dict:
                if pretrained_state_dict[key].dtype == torch.int64:
                    continue
                if pretrained_state_dict[key].dtype == torch.uint8:
                    continue
                self.theta[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.theta:
                if key not in other.theta:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.theta[key] + other.theta[key]
        return self.__class__(vector=new_vector)

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.theta:
                new_vector[key] = -self.theta[key]
        return self.__class__(vector=new_vector)

    def __pow__(self, power):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.theta:
                new_vector[key] = self.theta[key] ** power
        return self.__class__(vector=new_vector)

    def __mul__(self, other):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.theta:
                new_vector[key] = other * self.theta[key]
        return self.__class__(vector=new_vector)

    def dot(self, other):
        """Dot product of two task vectors."""
        with torch.no_grad():
            dot_product = 0.0
            for key in self.theta:
                if key not in other.theta:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.theta[key] * other.theta[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))


class MTLTaskVector(_TaskVector):
    def __init__(
        self,
        pretrained_checkpoint: _Checkpoint = None,
        finetuned_checkpoint: _Checkpoint = None,
        theta=None,
        tau=None,
    ):
        if theta is not None and tau is not None:
            super().__init__(None, None, theta)
            self.tau = tau
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_model = self._safe_load(pretrained_checkpoint)
                finetuned_model = self._safe_load(finetuned_checkpoint)
                self.head_tasks = finetuned_model.head_tasks

                # parse pretrained_checkpoint
                pretrained_state_dict = {
                    k: v for k, v in pretrained_model.state_dict().items()
                    if not any(task in k for task in pretrained_model.head_tasks)
                }

                # parse finetuned_checkpoint
                finetuned_state_dict = {}
                task_specific_state_dict = {}
                for k, v in finetuned_model.state_dict().items():
                    if any(task in k for task in self.head_tasks):
                        task_specific_state_dict[k] = v
                    else:
                        finetuned_state_dict[k] = v

                super().__init__(pretrained_state_dict, finetuned_state_dict, None)
                self.tau = task_specific_state_dict

    def _safe_load(self, checkpoint):
        if isinstance(checkpoint, str):
            return torch_load(checkpoint)
        elif isinstance(checkpoint, nn.Module):
            # Create a new model with the same architecture
            model_class_name = checkpoint.__class__.__name__
            new_model = globals()[model_class_name](head_tasks=checkpoint.head_tasks)
            new_model.load_state_dict(checkpoint.state_dict())
            return new_model
        else:
            raise ValueError(f"Invalid type for checkpoint: {type(checkpoint)}")

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pt_model = self._safe_load(pretrained_checkpoint)
            pt_model_state_dict = pt_model.state_dict()
            # updates = {}
            
            for param_name, param_value in self.theta.items():
                pt_model_state_dict[param_name].add_(scaling_coef * param_value)
                # updates[param_name]  = pt_model_state_dict[param_name] + scaling_coef * param_value
            
            # pt_model.load_state_dict({**pt_model.state_dict(), **updates, **self.tau})
            pt_model.load_state_dict({**pt_model.state_dict(), **self.tau}, strict=False)
        
        return pt_model
