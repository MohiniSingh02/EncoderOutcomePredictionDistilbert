from typing import Iterable

import torch
import wandb
from torch import tensor, Tensor
from torchmetrics.retrieval import RetrievalAUROC
from wandb import Table

top_ks = [1, 3, 5, 10, 20, 30, 50]


class MultilabelSkipAUROC(RetrievalAUROC):
    def __init__(self, num_labels):
        super().__init__(empty_target_action='skip')
        self.class_indices = torch.arange(0, num_labels)

    def update(self, preds: Tensor, target: Tensor, **kwargs) -> None:
        """Check shape, check and convert dtypes, flatten and add to accumulators.
        """
        return super().update(preds, target, self.class_indices.expand(preds.shape[0], -1))


def micro_and_macro(metric_cls, name, *args, at_x=False, **kwargs):
    metrics = {f'Micro{name}': metric_cls(*args, **kwargs, average='micro'),
               f'Macro{name}': metric_cls(*args, **kwargs, average='macro')}
    if at_x:
        metrics |= build_metric_at_x(metric_cls, f'Micro{name}', *args, **kwargs, average='micro')
        metrics |= build_metric_at_x(metric_cls, f'Macro{name}', *args, **kwargs, average='macro')
    return metrics


def build_metric_at_x(metric_cls, name, *args, **kwargs):
    return {f'{name}@{k}': metric_cls(*args, **kwargs, top_k=k) for k in top_ks}


def compute_thresholds(pr_curve_results: Iterable[tensor], out_tensor: tensor) -> (tensor, dict[str, tensor]):
    for i, (p, r, t) in enumerate(zip(*pr_curve_results)):
        # remove last entry which is just there for backwards compatibility
        f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1])
        f1 = torch.nan_to_num(f1, 0)
        max_f1, ix = torch.max(f1, dim=0)

        if max_f1 == 0:  # if there's no successful threshold, use  at least 0.5 or the biggest and then some
            out_tensor[i] = max(t[-1] + 1e-6, 0.5)
        elif ix == 0:  # if the first is the best use something a little lower or 0.5 if it's smaller
            out_tensor[i] = min(t[0] - 1e-6, 0.5)
        else:  # else take the middle between the best and the previous one
            out_tensor[i] = (t[ix - 1] + t[ix]) / 2

    return out_tensor


def log_multi_element_tensors_as_table(metrics: dict[str, tensor]):
    for k, v in metrics.items():
        if v.numel() > 1:
            if v.numel() == 5:
                wandb.log({k: Table(columns=['TP', 'FP', 'TN', 'FN', 'SUP'], data=[list(v)])})
                del metrics[k]
            else:
                raise NotImplementedError('Unknown amount of numels %d for metric %s!', v.numel(), k)
