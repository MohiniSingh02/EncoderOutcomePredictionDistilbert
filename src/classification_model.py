import re
from typing import Optional

import torch
import transformers
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.nn import BCEWithLogitsLoss
from torchmetrics import F1Score, MetricCollection, Recall, Precision, Accuracy
from torchmetrics.classification import MultilabelPrecisionRecallCurve, MultilabelAUROC
from torchmetrics.retrieval import RetrievalMAP
from transformers import BertModel

from metrics import build_metric_at_x, compute_all_metrics, aggregate_AUROC


def extract_re_group(input_string, pattern):
    match = re.search(pattern, input_string)
    return match.group(1) if match else 'not found'


class ClassificationModel(LightningModule):
    def __init__(self,
                 num_classes: int = 1446,
                 encoder_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 warmup_steps: int = 0,
                 decay_steps: int = 50_000,
                 weight_decay: float = 0.01,
                 lr: float = 2e-5,
                 optimizer_name="adam",
                 ):
        super().__init__()
        self.save_hyperparameters({
            'icd': extract_re_group(self.val_dataloader().dataset.data_dir, r'icd-?(\d{1,2})'),
            'split': extract_re_group(self.val_dataloader().dataset.data_dir, r'(icu|hosp)')
        })

        self.encoder = BertModel.from_pretrained(encoder_model_name)
        self.encoder.pooler = None
        self.num_classes = num_classes
        self.classification_layer = torch.nn.Linear(768, self.num_classes)

        metrics = MetricCollection(
            build_metric_at_x(RetrievalMAP, 'mAP') |
            {'AUROC': MultilabelAUROC(num_labels=self.num_classes, average=None),
             'PRCurve': MultilabelPrecisionRecallCurve(num_labels=self.num_classes)
             }
        )
        self.test_metrics = metrics.clone('Test/')
        self.val_metrics = metrics.clone('Val/')

        main_metrics = MetricCollection(
            build_metric_at_x(Recall, 'Recall', 'multiclass', num_classes=self.num_classes, micro=True, macro=True) |
            build_metric_at_x(Precision, 'Precision', 'multiclass', num_classes=self.num_classes, micro=True, macro=True) |
            build_metric_at_x(F1Score, 'F1', 'multiclass', num_classes=self.num_classes, micro=True, macro=True) |
            build_metric_at_x(Accuracy, 'Accuracy', 'multiclass', num_classes=self.num_classes, micro=True, macro=True),
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
        tensor_labels = torch.concat(labels)

        auroc_name = metrics.prefix + 'AUROC'
        prc_name = metrics.prefix + 'PRCurve'

        metrics_dict = metrics.compute()
        metrics_dict[auroc_name] = aggregate_AUROC(metrics_dict[auroc_name], tensor_labels)

        preds = torch.sigmoid(torch.concat(logits))
        metrics_dict |= compute_all_metrics(metrics_dict[prc_name], preds, tensor_labels, metrics.prefix)

        # Remove PRCurve data, since it can't be logged easily
        del metrics_dict[prc_name]
        self.logger.log_metrics({k: v.float() for k, v in metrics_dict.items()}, self.global_step)

        main_metrics_dict = main_metrics.compute()
        self.logger.log_metrics(main_metrics_dict, self.global_step)

        metrics.reset()
        main_metrics.reset()
        logits.clear()
        labels.clear()

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
