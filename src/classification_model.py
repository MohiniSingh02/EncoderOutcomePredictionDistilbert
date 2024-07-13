from typing import Optional, Literal

import torch
import transformers
from torch import tensor
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.nn import BCEWithLogitsLoss
from torchmetrics import F1Score, MetricCollection, Recall, Precision, Accuracy
from torchmetrics.classification import MultilabelPrecisionRecallCurve, MultilabelAUROC
from torchmetrics.functional.classification import multilabel_stat_scores
from torchmetrics.retrieval import RetrievalMAP
from transformers import BertModel


class ClassificationModel(LightningModule):
    def __init__(self,
                 num_classes: int = 1446,
                 encoder_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 warmup_steps: int = 0,
                 decay_steps: int = 50_000,
                 weight_decay: float = 0.01,
                 lr: float = 2e-5,
                 optimizer_name="adam",
                 readmission_task: bool = False
                 ):
        super().__init__()
        self.encoder = BertModel.from_pretrained(encoder_model_name)
        self.encoder.pooler = None
        self.num_classes = num_classes
        self.classification_layer = torch.nn.Linear(768, self.num_classes)

        # Precision / Recall / AUROC
        # multilabel
        # Precision / Recall @ 20, 10, 5, 3, 1
        # Precision / Recall / Accuracy @ first
        # diagnosis @ 5, 3, 1
        # Retrieval
        # mAP

        metrics = MetricCollection(
            self.build_metric_at_x(RetrievalMAP, 'mAP') |
            {'AUROC': MultilabelAUROC(num_labels=self.num_classes, average=None),
             'PRCurve': MultilabelPrecisionRecallCurve(num_labels=self.num_classes)
             }
        )
        self.test_metrics = metrics.clone('Test/')
        self.val_metrics = metrics.clone('Val/')

        main_metrics = MetricCollection(
            self.build_metric_at_x(Recall, 'Recall', 'multiclass', num_classes=self.num_classes, micro=True, macro=True) |
            self.build_metric_at_x(Precision, 'Precision', 'multiclass', num_classes=self.num_classes, micro=True, macro=True) |
            self.build_metric_at_x(F1Score, 'F1', 'multiclass', num_classes=self.num_classes, micro=True, macro=True) |
            self.build_metric_at_x(Accuracy, 'Accuracy', 'multiclass', num_classes=self.num_classes, micro=True, macro=True),
            postfix='_Main'
        )
        self.main_test_metrics = main_metrics.clone('Test/')
        self.main_val_metrics = main_metrics.clone('Val/')

        self.test_preds, self.test_labels = [], []
        self.val_logits, self.val_labels = [], []

        self.loss = BCEWithLogitsLoss()

        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.readmission_task = readmission_task

    def forward(self,
                input_ids,
                attention_mask):
        encoded = self.encoder(input_ids, attention_mask, return_dict=True)['last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss(logits, batch['labels'])
        self.log("Train/Loss", loss)
        return loss

    def test_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss(logits, batch['labels'])
        self.log("Test/Loss", loss)
        self.test_metrics.update(logits, batch['labels'].long(), indexes=batch['query_idces'])
        self.main_test_metrics.update(logits, batch['first_codes'])
        self.test_logits.append(logits)
        self.test_labels.append(batch['labels'])
        return loss

    def on_test_epoch_end(self) -> None:
        self.test_val_epoch_end(self.test_metrics, self.main_test_metrics, self.test_logits, self.test_labels)

    def validation_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss(logits, batch['labels'])
        self.log("Val/Loss", loss)
        self.val_metrics.update(logits, batch['labels'].long(), indexes=batch['query_idces'])
        self.main_val_metrics.update(logits, batch['first_codes'])
        self.val_logits.append(logits)
        self.val_labels.append(batch['labels'])
        return loss

    def on_validation_epoch_end(self) -> None:
        self.test_val_epoch_end(self.val_metrics, self.main_val_metrics, self.val_logits, self.val_labels)

    def test_val_epoch_end(self, metrics: MetricCollection, main_metrics, logits, labels) -> None:
        auroc_name = metrics.prefix + 'AUROC'
        prc_name = metrics.prefix + 'PRCurve'

        metrics_dict = metrics.compute()
        metrics_dict[auroc_name] = self.aggregate_AUROC(metrics_dict[auroc_name])

        preds = torch.sigmoid(torch.concat(logits))
        metrics_dict |= self.compute_all_metrics(metrics_dict[prc_name], preds, torch.concat(labels), metrics.prefix)

        # Remove PRCurve data, since it can't be logged easily
        self.logger.log_metrics(metrics_dict, self.global_step)

        main_metrics_dict = main_metrics.compute()
        self.logger.log_metrics(main_metrics_dict, self.global_step)

        metrics.reset()
        main_metrics.reset()

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        weight_decay = 0.01
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                weight_decay
        }, {
            'params':
                [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=weight_decay)

        scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, self.warmup_steps,
                                                                           self.decay_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
        }

        return [optimizer], [scheduler]

    top_ks = [1, 3, 5, 10, 20, 30, 50]

    def build_metric_at_x(self, metric_cls, name, *args, micro: bool = False, macro: bool = False, **kwargs):
        if not micro and not macro:
            return {name: metric_cls(*args, **kwargs)} | {f'{name}@{k}': metric_cls(*args, **kwargs, top_k=k) for k in self.top_ks}
        else:
            metrics = {}

            if micro:
                metrics |= self.build_metric_at_x(metric_cls, f'Micro{name}', *args, **kwargs, average='micro')
            if macro:
                metrics |= self.build_metric_at_x(metric_cls, f'Macro{name}', *args, **kwargs, average='macro')

            return metrics


    def compute_thresholds_and_tuned_macro(self, pr_curve_results: tensor, num_labels: int) -> (tensor, dict[str, tensor]):
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


    def compute_metrics(self, predictions: tensor, labels: tensor, prefix: str = '', post_fix: str = '',
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

    def compute_top_k(self, predictions: tensor, labels: tensor, prefix=''):
        batch_indices = torch.arange(predictions.shape[0]).unsqueeze(1).expand(-1, max(self.top_ks))
        preds_sorted, preds_sort_indices = predictions.sort(descending=True)
        predictions = predictions >= 0.5
        top_k_preds = torch.zeros_like(predictions)
        top_k_labels = torch.zeros_like(labels)

        metrics = {}
        previous_k = 0
        for k in self.top_ks:
            next_batches = batch_indices[:, previous_k:k]
            next_preds = preds_sort_indices[:, previous_k:k]
            top_k_preds[next_batches, next_preds] = predictions[next_batches, next_preds]
            top_k_labels[next_batches, next_preds] = labels[next_batches, next_preds]

            metrics |= self.compute_metrics(top_k_preds, top_k_labels, prefix + 'Micro', post_fix=f'@{k}', average='micro')
            metrics |= self.compute_metrics(top_k_preds, top_k_labels, prefix + 'Macro', post_fix=f'@{k}', average='macro')

            previous_k = k

        return metrics


    def compute_all_metrics(self, pr_curve_results, preds, labels, prefix: str):
        # Tuned metrics
        thresholds, metrics = self.compute_thresholds_and_tuned_macro(pr_curve_results, labels.shape[-1])
        metrics |= self.compute_metrics(preds >= thresholds, labels, 'TunedMicro', average='micro')
        scaled_preds = preds * 0.5 / thresholds
        metrics |= self.compute_top_k(scaled_preds, labels, 'Tuned')

        # Not tuned metrics
        metrics |= self.compute_metrics(preds, labels, 'Micro', average='micro')
        metrics |= self.compute_metrics(preds, labels, 'Macro', average='macro')
        metrics |= self.compute_top_k(preds, labels)

        return {prefix + k: v for k, v in metrics.items()}


    def aggregate_AUROC(self, auroc: tensor):
        return auroc[auroc > 0].mean()
