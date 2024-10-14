import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from copy import deepcopy
from PIL import Image
from scipy.optimize import minimize


"""
Define task metrics, loss functions and model trainer here.
"""
class ConfMatrix(object):
    """
    For mIoU and other pixel-level classification tasks.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def reset(self):
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item()



"""
Define TaskMetric class to record task-specific metrics.
"""
class TaskMetric:
    def __init__(self, train_tasks, pri_tasks, batch_size, epochs, dataset, include_mtl=False):
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks
        self.batch_size = batch_size
        self.dataset = dataset
        self.include_mtl = include_mtl
        self.metric = {key: np.zeros([epochs, 2]) for key in train_tasks.keys()}  # record loss & task-specific metric
        self.data_counter = 0
        self.epoch_counter = 0
        self.conf_mtx = {}

        if include_mtl:  # include multi-task performance (relative averaged task improvement)
            self.metric['all'] = np.zeros(epochs)
        
        for task in self.train_tasks:
            if task in ['seg', 'part_seg']:
                self.conf_mtx[task] = ConfMatrix(self.train_tasks[task]["num_classes"])

    def reset(self):
        """
        Reset data counter and confusion matrices.
        """
        self.epoch_counter += 1
        self.data_counter = 0

        if len(self.conf_mtx) > 0:
            for i in self.conf_mtx:
                self.conf_mtx[i].reset()

    def update_metric(self, train_res, task_gt):
        """
        Update batch-wise metric for each task.
            :param task_pred: [TASK_PRED1, TASK_PRED2, ...]
            :param task_gt: {'TASK_ID1': TASK_GT1, 'TASK_ID2': TASK_GT2, ...}
            :param task_loss: [TASK_LOSS1, TASK_LOSS2, ...]
        """
        curr_bs = list(train_res.values())[0]["pred"][0].shape[0]
        r = self.data_counter / (self.data_counter + curr_bs / self.batch_size)
        e = self.epoch_counter
        self.data_counter += 1

        with torch.no_grad():
            for task_id, gt in task_gt.items():
                pred = train_res[task_id]["pred"]
                self.metric[task_id][e, 0] = r * self.metric[task_id][e, 0] + (1 - r) * train_res[task_id]["total_loss"].item()

                if task_id in ['seg', 'part_seg']:
                    # update confusion matrix (metric will be computed directly in the Confusion Matrix)
                    self.conf_mtx[task_id].update(pred.argmax(1).flatten(), gt.flatten())

                if 'class' in task_id:
                    # Accuracy for image classification tasks
                    pred_label = pred.data.max(1)[1]
                    acc = pred_label.eq(gt).sum().item() / pred_label.shape[0]
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * acc

                if task_id in ['depth', 'disp', 'noise']:
                    # Abs. Err.
                    invalid_idx = -1 if task_id == 'disp' else 0
                    valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
                    abs_err = torch.mean(torch.abs(pred - gt).masked_select(valid_mask)).item()
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * abs_err

                if task_id in ['normal']:
                    # Mean Degree Err.
                    valid_mask = (torch.sum(gt, dim=1) != 0).to(pred.device)
                    degree_error = torch.acos(torch.clamp(torch.sum(pred * gt, dim=1).masked_select(valid_mask), -1, 1))
                    mean_error = torch.mean(torch.rad2deg(degree_error)).item()
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * mean_error

    def compute_metric(self, only_pri=False):
        metric_str = ''
        e = self.epoch_counter
        tasks = self.pri_tasks if only_pri else self.train_tasks  # only print primary tasks performance in evaluation

        for task_id in tasks:
            if task_id in ['seg', 'part_seg']:  # mIoU for segmentation
                self.metric[task_id][e, 1] = self.conf_mtx[task_id].get_metrics()

            metric_str += ' {} {:.4f} {:.4f}'\
                .format(task_id.capitalize(), self.metric[task_id][e, 0], self.metric[task_id][e, 1])

        if self.include_mtl:
            # Pre-computed single task learning performance using trainer_dense_single.py
            if self.dataset == 'nyuv2':
                stl = {'seg': 0.4337, 'depth': 0.5224, 'normal': 22.40}
            elif self.dataset == 'cityscapes':
                stl = {'seg': 0.5620, 'part_seg': 0.5274, 'disp': 0.84}

            delta_mtl = 0
            for task_id in self.train_tasks:
                if task_id in ['seg', 'part_seg'] or 'class' in task_id:  # higher better
                    delta_mtl += (self.metric[task_id][e, 1] - stl[task_id]) / stl[task_id]
                elif task_id in ['depth', 'normal', 'disp']:
                    delta_mtl -= (self.metric[task_id][e, 1] - stl[task_id]) / stl[task_id]

            self.metric['all'][e] = delta_mtl / len(stl)
            metric_str += ' | All {:.4f}'.format(self.metric['all'][e])
        return metric_str
    
    def get_metric(self, task):
        return self.metric[task][self.epoch_counter-1, 1]

    def get_best_performance(self, task):
        e = self.epoch_counter
        if task in ['seg', 'part_seg'] or 'class' in task:  # higher better
            return max(self.metric[task][:e, 1])
        if task in ['depth', 'normal', 'disp']:  # lower better
            return min(self.metric[task][:e, 1])
        if task in ['all']:  # higher better
            return max(self.metric[task][:e])



"""
Define Gradient-based frameworks here. 
Based on https://github.com/Cranial-XIX/CAGrad/blob/main/cityscapes/utils.py
"""
def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
    U = torch.rand_like(grads[:, 0])
    M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g


def pcgrad(grads, rng, num_tasks):
    grad_vec = grads.t()

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (grad_vec.norm(dim=1, keepdim=True) + 1e-8)  # num_tasks x dim
    modified_grad_vec = deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[task_indices]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(dim=1, keepdim=True)   # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g


def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(
            x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.shared_modules():
        for p in mm.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1


def overwrite_grad(m, newgrad, grad_dims, num_tasks):
    newgrad = newgrad * num_tasks  # to match the sum loss
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1
        


"""
Visualize predictions for semantic segmentation and depth estimation tasks.
"""

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def visualize_semantic_classes(epoch, original_image, pred_seg, target_seg, alpha=0.4):
    n_classes = pred_seg.shape[1]
    pred_seg = pred_seg.argmax(1)

    for idx in range(pred_seg.shape[0]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(normalize(original_image[idx].permute(1, 2, 0).cpu().numpy()))
        ax[0].imshow(target_seg[idx].cpu().numpy(), cmap='tab20', alpha=alpha, vmin=0, vmax=n_classes-1)
        ax[0].set_title('Ground Truth')
        ax[0].axis('off')

        ax[1].imshow(normalize(original_image[idx].permute(1, 2, 0).cpu().numpy()))
        ax[1].imshow(pred_seg[idx].cpu().numpy(), cmap='tab20', alpha=alpha, vmin=0, vmax=n_classes-1)
        ax[1].set_title('Prediction')
        ax[1].axis('off')

        # Save the figure to a PIL Image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        wandb.log({f"seg_task_image/epoch_{epoch}": wandb.Image(pil_image)})
        plt.close(fig)


def visualize_depth(epoch, original_image, pred_depth, target_depth, alpha=0.4):
    for idx in range(pred_depth.shape[0]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Target Depth
        ax[0].imshow(normalize(original_image[idx].permute(1, 2, 0).cpu().numpy()))
        ax[0].imshow(target_depth[idx].squeeze(0).cpu().numpy(), cmap='jet', alpha=alpha)
        ax[0].set_title('Target Depth')
        ax[0].axis('off')

        # Predicted Depth
        ax[1].imshow(normalize(original_image[idx].permute(1, 2, 0).cpu().numpy()))
        ax[1].imshow(pred_depth[idx].squeeze(0).cpu().numpy(), cmap='jet', alpha=alpha)
        ax[1].set_title('Predicted Depth')
        ax[1].axis('off')

        # Save the figure to a PIL Image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        wandb.log({f"depth_task_image/epoch_{epoch}": wandb.Image(pil_image)})
        plt.close(fig)
        

# normal vector to rgb values
def norm_to_rgb(norm):
    norm_rgb = (norm + 1.0) * 0.5
    norm_rgb = np.clip(norm_rgb * 255.0, 0, 255)

def visualize_normals(epoch, original_image, pred_normal, target_normal):
    for idx in range(pred_normal.shape[0]):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].imshow(normalize(original_image[idx].permute(1, 2, 0).cpu().numpy()))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(normalize(target_normal[idx].permute(1, 2, 0).cpu().numpy()),  cmap='jet')
        ax[1].set_title('Target Normal')
        ax[1].axis('off')
        
        ax[2].imshow(normalize(pred_normal[idx].permute(1, 2, 0).cpu().numpy()),  cmap='jet')
        ax[2].set_title('Predicted Normal')
        ax[2].axis('off')
        
        # Save the figure to a PIL Image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        wandb.log({f"normal_task_image/epoch_{epoch}": wandb.Image(pil_image)})
        plt.close(fig)
        


VISUALIZATION_FUNCS = {
    'seg': visualize_semantic_classes,
    'depth': visualize_depth,
    'normal': visualize_normals,
}



"""
Define model evaluation function.
"""
def compute_loss(pred, gt, task_id):
    """
    Compute task-specific loss.
    """
    if task_id in ['seg', 'part_seg'] or 'class' in task_id:
        # Cross Entropy Loss with Ignored Index (values are -1)
        loss = F.cross_entropy(pred, gt, ignore_index=-1)

    if task_id in ['normal', 'depth', 'disp', 'noise']:
        # L1 Loss with Ignored Region (values are 0 or -1)
        invalid_idx = -1 if task_id == 'disp' else 0
        valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
        loss = torch.sum(F.l1_loss(pred, gt, reduction='none').masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)
    return loss

def eval(
    epoch: int,
    model, 
    data_loader: torch.utils.data.DataLoader,
    test_metric: TaskMetric,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    mode: str = "head",
):
    data_batch = len(data_loader)
    # TODO: switch to random batch
    # viz_batch_idx = np.random.randint(0, data_batch)
    viz_batch_idx = 0

    model.eval()
    with torch.no_grad():
        dataset = iter(data_loader)
        for batch_idx in range(data_batch):
            img, target = next(dataset)
            img = img.to(device)
            target = {task_id: target[task_id].to(device) for task_id in model.head_tasks.keys()}

            test_res = model(img, None, img_gt=target, return_loss=True)
            
            test_metric.update_metric(test_res, target)

            if batch_idx == viz_batch_idx:
                for task_id in model.head_tasks:
                    if task_id in VISUALIZATION_FUNCS:
                        VISUALIZATION_FUNCS[task_id](epoch, img, test_res[task_id]["pred"], target[task_id])

    test_str = test_metric.compute_metric()
    test_metric.reset()

    wandb.log({
        **{f"{mode}/test/loss/{task_id}": test_res[task_id]["total_loss"] for task_id in model.head_tasks},
        **{f"{mode}/test/metric/{task_id}": test_metric.get_metric(task_id) for task_id in model.head_tasks}
    },) # step=epoch

    return test_str



"""
Define task flags and weight strings.
"""
def create_task_flags(task, dataset, with_noise=False):
    """
    Record task and its prediction dimension.
    Noise prediction is only applied in auxiliary learning.
    """
    nyu_tasks = {
        "seg": {
            "num_classes": 13,
        },
        "depth": {
            "num_classes": 1,
            "min_depth": 0.0,
            "max_depth": 10.0,
        },
        'normal': {
            'num_classes': 3,
        },
    }
    cityscapes_tasks = {'seg': 19, 'part_seg': 10, 'disp': 1}  # TODO
    dataset_tasks = {
        'nyuv2': nyu_tasks,
        'cityscapes': cityscapes_tasks
    }

    tasks = dataset_tasks.get(dataset, {})
    if task != 'all':
        tasks = {task: tasks.get(task)}

    if with_noise:
        tasks['noise'] = {'num_classes': 1}

    return tasks


def get_weight_str(weight, tasks):
    """
    Record task weighting.
    """
    weight_str = 'Task Weighting | '
    for i, task_id in enumerate(tasks):
        weight_str += '{} {:.04f} '.format(task_id.title(), weight[i])
    return weight_str


def get_weight_str_ranked(weight, tasks, rank_num):
    """
    Record top-k ranked task weighting.
    """
    rank_idx = np.argsort(weight)

    if type(tasks) == dict:
        tasks = list(tasks.keys())

    top_str = 'Top {}: '.format(rank_num)
    bot_str = 'Bottom {}: '.format(rank_num)
    for i in range(rank_num):
        top_str += '{} {:.02f} '.format(tasks[rank_idx[-i-1]].title(), weight[rank_idx[-i-1]])
        bot_str += '{} {:.02f} '.format(tasks[rank_idx[i]].title(), weight[rank_idx[i]])

    return 'Task Weighting | {}| {}'.format(top_str, bot_str)