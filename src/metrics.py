from typing import Optional, Literal

import torch
from torch import tensor
from torchmetrics.functional.classification import multilabel_stat_scores, multilabel_precision_recall_curve

top_ks = [1, 3, 5, 10, 20, 30, 50]


def build_metric_at_x(metric_cls, name, *args, micro: bool = False, macro: bool = False, **kwargs):
    if not micro and not macro:
        return {name: metric_cls(*args, **kwargs)} | {f'{name}@{k}': metric_cls(*args, **kwargs, top_k=k) for k in top_ks}
    else:
        metrics = {}

        if micro:
            metrics |= build_metric_at_x(metric_cls, f'Micro{name}', *args, **kwargs, average='micro')
        if macro:
            metrics |= build_metric_at_x(metric_cls, f'Macro{name}', *args, **kwargs, average='macro')

        return metrics


def compute_thresholds_and_tuned_macro(pr_curve_results: tensor, num_labels: int) -> (tensor, dict[str, tensor]):
    precision_per_class = torch.empty(num_labels)
    recall_per_class = torch.empty(num_labels)
    f1_per_class = torch.empty(num_labels)
    thresholds = torch.empty(num_labels)

    for i, (p, r, t) in enumerate(zip(*pr_curve_results)):
        f1 = 2 * p * r / (p + r)
        f1 = torch.nan_to_num(f1, 0)
        ix = torch.argmax(f1)

        recall_per_class[i] = r[ix]
        precision_per_class[i] = p[ix]
        f1_per_class[i] = f1[ix]
        thresholds[i] = t[ix]

    return thresholds, {
        'TunedMacroPrecision': precision_per_class.mean(),
        'TunedMacroRecall': recall_per_class.mean(),
        'TunedMacroF1': f1_per_class.mean()
    }


def compute_metrics(predictions: tensor, labels: tensor, prefix: str = '', post_fix: str = '',
                    average: Optional[Literal["micro", "macro", "weighted", "none"]] = 'micro'):
    if average == 'macro':
        tp, fp, tn, fn, sup = multilabel_stat_scores(predictions, labels, num_labels=labels.shape[-1], average=None).T
    else:
        tp, fp, tn, fn, sup = multilabel_stat_scores(predictions, labels, num_labels=labels.shape[-1], average=average)

    precision = torch.nan_to_num(tp / (tp + fp), 0)
    recall = torch.nan_to_num(tp / (tp + fn))
    f1 = torch.nan_to_num(2 * precision * recall / (precision + recall))
    results = {
        f'{prefix}Precision{post_fix}': precision,
        f'{prefix}Recall{post_fix}': recall,
        f'{prefix}F1{post_fix}': f1,
        f'{prefix}TP{post_fix}': tp,
        f'{prefix}FP{post_fix}': fp,
        f'{prefix}TN{post_fix}': tn,
        f'{prefix}FN{post_fix}': fn,
        f'{prefix}SUP{post_fix}': sup
    }
    if average == 'macro':
        return {k: v.float().mean() for k, v in results.items()}
    else:
        return results


# tune logits:
# 0.5 / thresholds * logits

def compute_top_k(predictions: tensor, labels: tensor, prefix=''):
    batch_indices = torch.arange(predictions.shape[0]).unsqueeze(1).expand(-1, max(top_ks))
    preds_sorted, preds_sort_indices = predictions.sort(descending=True)
    predictions = predictions >= 0.5
    top_k_preds = torch.zeros_like(predictions)
    top_k_labels = torch.zeros_like(labels)

    metrics = {}
    previous_k = 0
    for k in top_ks:
        next_batches = batch_indices[:, previous_k:k]
        next_preds = preds_sort_indices[:, previous_k:k]
        top_k_preds[next_batches, next_preds] = predictions[next_batches, next_preds]
        top_k_labels[next_batches, next_preds] = labels[next_batches, next_preds]

        metrics |= compute_metrics(top_k_preds, top_k_labels, prefix + 'Micro', post_fix=f'@{k}', average='micro')
        metrics |= compute_metrics(top_k_preds, top_k_labels, prefix + 'Macro', post_fix=f'@{k}', average='macro')

        previous_k = k

    return metrics


def compute_all_metrics(pr_curve_results, preds, labels, prefix: str):
    # Tuned metrics
    thresholds, metrics = compute_thresholds_and_tuned_macro(pr_curve_results, labels.shape[-1])
    metrics |= compute_metrics(preds >= thresholds, labels, 'TunedMicro', average='micro')
    scaled_preds = preds * 0.5 / thresholds
    metrics |= compute_top_k(scaled_preds, labels, 'Tuned')

    # Not tuned metrics
    metrics |= compute_metrics(preds, labels, 'Micro', average='micro')
    metrics |= compute_metrics(preds, labels, 'Macro', average='macro')
    metrics |= compute_top_k(preds, labels)

    return {prefix + k: v for k, v in metrics.items()}


def aggregate_AUROC(auroc: tensor):
    return auroc[auroc > 0].mean()


if __name__ == '__main__':
    # num_labels = 3
    # logits = torch.logit(torch.tensor([[0.75, 0.05, 0.35],
    #                                    [0.45, 0.75, 0.05],
    #                                    [0.05, 0.55, 0.75],
    #                                    [0.05, 0.65, 0.05]]))
    # labels = torch.tensor([[1, 0, 1],
    #                        [0, 0, 0],
    #                        [0, 1, 1],
    #                        [1, 1, 1]])

    samples = 20
    num_labels = 50
    logits = torch.randn([samples, num_labels])
    labels = torch.rand([samples, num_labels]) > 0.5

    preds = torch.sigmoid(logits)
    # labels = preds > 0.3

    pr_curve_results = multilabel_precision_recall_curve(preds, labels, num_labels=num_labels)
    compute_all_metrics(pr_curve_results, preds, labels, 'bla/')

    print('done')
