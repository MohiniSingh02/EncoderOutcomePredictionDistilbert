import re
from typing import Optional

import torch
import transformers
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import tensor, sigmoid
from torchmetrics import F1Score, MetricCollection, Recall, Precision, Accuracy
from torchmetrics.classification import MultilabelPrecisionRecallCurve, MultilabelStatScores
from torchmetrics.retrieval import RetrievalMAP, RetrievalPrecision, RetrievalRecall, RetrievalAUROC

from metrics import build_metric_at_x, compute_thresholds, MultilabelSkipAUROC
from src.bert_model import BertForSequenceClassificationWithoutPooling
from src.metrics import stat_metrics_to_table


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
        self.save_hyperparameters({'num_classes': num_classes})

        self.encoder = BertForSequenceClassificationWithoutPooling.from_pretrained(encoder_model_name)
        self.num_classes = num_classes

        self.pr_curve = MultilabelPrecisionRecallCurve(num_labels=self.num_classes)
        metrics = self.create_metrics()
        self.test_metrics = metrics.clone('Test/')
        self.tuned_test_metrics = metrics.clone('Test/Tuned')
        self.val_metrics = metrics.clone('Val/')
        self.tuned_val_metrics = metrics.clone('Val/Tuned')

        main_metrics = self.create_main_diagnosis_metrics()
        self.main_test_metrics = main_metrics.clone('Test/')
        self.main_val_metrics = main_metrics.clone('Val/')

        self.test_preds, self.test_labels = [], []
        self.val_preds, self.val_labels = [], []

        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr = lr

    def create_metrics(self):
        return MetricCollection(
            build_metric_at_x(RetrievalMAP, 'mAP') |
            build_metric_at_x(RetrievalRecall, 'Recall') |
            build_metric_at_x(RetrievalPrecision, 'Precision') |
            build_metric_at_x(RetrievalAUROC, 'AUROC', empty_target_action='skip') |
            {'AUROC': MultilabelSkipAUROC(num_labels=self.num_classes),
             'MacroStats': MultilabelStatScores(num_labels=self.num_classes, average='macro'),
             'MicroStats': MultilabelStatScores(num_labels=self.num_classes, average='micro')},
        )

    def create_main_diagnosis_metrics(self):
        return MetricCollection(
            build_metric_at_x(Recall, 'Recall', 'multiclass', num_classes=self.num_classes, micro=True, macro=True) |
            build_metric_at_x(Precision, 'Precision', 'multiclass', num_classes=self.num_classes, micro=True,
                              macro=True) |
            build_metric_at_x(F1Score, 'F1', 'multiclass', num_classes=self.num_classes, micro=True, macro=True) |
            build_metric_at_x(Accuracy, 'Accuracy', 'multiclass', num_classes=self.num_classes, micro=True, macro=True),
            postfix='_Main'
        )

    def setup(self, **kwargs):
        if self.trainer is not None:
            checkpoint_callback = self.trainer.checkpoint_callback
            if checkpoint_callback:
                checkpoint_callback.CHECKPOINT_NAME_LAST = 'lastckpt_' + checkpoint_callback.filename
                print(checkpoint_callback.CHECKPOINT_NAME_LAST)

            if self.trainer.datamodule is not None:
                data_module = self.trainer.datamodule
                self.save_hyperparameters({
                    'icd': extract_re_group(str(data_module.data_dir), r'icd-?(\d{1,2})'),
                    'split': extract_re_group(str(data_module.data_dir), r'(icu|hosp)')
                })

    def forward(self, input_ids, attention_mask):
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

        preds = sigmoid(logits)
        targets = batch['labels'].long()

        self.test_metrics.update(preds, targets, indexes=batch['query_idces'])
        self.main_test_metrics.update(preds, batch['first_codes'])

        self.test_preds.append(preds)
        self.test_labels.append(targets)

        return loss

    def validation_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        logits = self(batch['input_ids'], batch['attention_mask'])

        loss = self.loss(logits, batch['labels'])
        self.log("Val/Loss", loss)

        preds = sigmoid(logits)
        targets = batch['labels'].long()

        self.pr_curve.update(preds, targets)
        self.val_metrics.update(preds, targets, indexes=batch['query_idces'])
        self.main_val_metrics.update(preds, batch['first_codes'])

        self.val_preds.append(preds)
        self.val_labels.append(targets)

        return loss

    def on_test_epoch_end(self) -> None:
        self.test_val_epoch_end(self.test_metrics, self.main_test_metrics, self.tuned_test_metrics,
                                self.test_preds, self.test_labels)

    def on_validation_epoch_end(self) -> None:
        self.encoder.tuning_weights = torch.logit(compute_thresholds(self.pr_curve.compute(), self.thresholds))
        self.test_val_epoch_end(self.val_metrics, self.main_val_metrics, self.tuned_val_metrics,
                                self.val_preds, self.val_labels)

    def test_val_epoch_end(self, metrics: MetricCollection, main_metrics: MetricCollection,
                           tuned_metrics: MetricCollection, preds: list[tensor], labels: list[tensor]) -> None:
        scaled_preds = torch.concat(preds) * 0.5 / self.thresholds
        indexes = torch.arange(len(scaled_preds)).unsqueeze(1).expand(len(scaled_preds), self.num_classes)
        metrics_dict = metrics.compute() | main_metrics.compute() | tuned_metrics(scaled_preds, torch.concat(labels),
                                                                                  indexes=indexes)
        stat_metrics_to_table(metrics_dict, metrics.prefix)
        self.log_dict(metrics_dict)

        metrics.reset()
        tuned_metrics.reset()
        main_metrics.reset()

        preds.clear()
        labels.clear()

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                self.weight_decay
        }, {
            'params':
                [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay)

        scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, self.warmup_steps,
                                                                           self.decay_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
        }

        return [optimizer], [scheduler]
