import torch
import wandb
from torch import tensor, Tensor
from torchmetrics import MetricCollection, Recall, Precision, F1Score, Accuracy
from torchmetrics.classification import MultilabelStatScores
from torchmetrics.retrieval import RetrievalAUROC, RetrievalMAP, RetrievalRecall, RetrievalPrecision
from wandb import Table

top_ks = [1, 3, 5, 10, 20, 30, 50]


class MultilabelSkipAUROC(RetrievalAUROC):
    def __init__(self, num_labels):
        super().__init__(empty_target_action='skip')
        self.register_buffer('class_indices', torch.arange(0, num_labels))

    def update(self, preds: Tensor, target: Tensor, **kwargs) -> None:
        """Check shape, check and convert dtypes, flatten and add to accumulators.
        """
        return super().update(preds, target, self.class_indices.expand(preds.shape[0], -1))


def create_metrics(num_classes):
    return MetricCollection(
        build_metric_at_x(RetrievalMAP, 'mAP') |
        build_metric_at_x(RetrievalRecall, 'Recall') |
        build_metric_at_x(RetrievalPrecision, 'Precision') |
        build_metric_at_x(RetrievalAUROC, 'AUROC', empty_target_action='skip') |

        micro_and_macro(Recall, 'Recall', 'multilabel', num_labels=num_classes) |
        micro_and_macro(Precision, 'Precision', 'multilabel', num_labels=num_classes) |
        micro_and_macro(F1Score, 'F1', 'multilabel', num_labels=num_classes) |
        micro_and_macro(Accuracy, 'Accuracy', 'multilabel', num_labels=num_classes) |

        {'AUROC': MultilabelSkipAUROC(num_labels=num_classes),
         'MacroStats': MultilabelStatScores(num_labels=num_classes, average='macro'),
         'MicroStats': MultilabelStatScores(num_labels=num_classes, average='micro')},
    )


def create_main_diagnosis_metrics(num_classes):
    return MetricCollection(
        micro_and_macro(Recall, 'Recall', 'multiclass', num_classes=num_classes, at_x=True) |
        micro_and_macro(Precision, 'Precision', 'multiclass', num_classes=num_classes, at_x=True) |
        micro_and_macro(F1Score, 'F1', 'multiclass', num_classes=num_classes, at_x=True) |
        micro_and_macro(Accuracy, 'Accuracy', 'multiclass', num_classes=num_classes, at_x=True),
        postfix='_Main'
    )



def micro_and_macro(metric_cls, name, *args, at_x=False, **kwargs):
    metrics = {f'Micro{name}': metric_cls(*args, **kwargs, average='micro'),
               f'Macro{name}': metric_cls(*args, **kwargs, average='macro')}
    if at_x:
        metrics |= build_metric_at_x(metric_cls, f'Micro{name}', *args, **kwargs, average='micro')
        metrics |= build_metric_at_x(metric_cls, f'Macro{name}', *args, **kwargs, average='macro')
    return metrics


def build_metric_at_x(metric_cls, name, *args, **kwargs):
    return {f'{name}@{k}': metric_cls(*args, **kwargs, top_k=k) for k in top_ks}

def merge_and_reset_metrics(*metrics: MetricCollection) -> dict[str, torch.tensor]:
    metrics_dict = {}

    for metric in metrics:
        computed_metric = metric.compute()
        log_multi_element_tensors_as_table(computed_metric)
        metrics_dict |= computed_metric
        metric.reset()

    return metrics_dict


def log_multi_element_tensors_as_table(metrics: dict[str, tensor]):
    to_delete = []
    for k, v in metrics.items():
        if v.numel() > 1:
            if v.numel() == 5:
                wandb.log({k: Table(columns=['TP', 'FP', 'TN', 'FN', 'SUP'], data=[list(v)])})
                to_delete.append(k)
            else:
                raise NotImplementedError('Unknown amount of numels %d for metric %s!', v.numel(), k)
    for k in to_delete:
        del metrics[k]
