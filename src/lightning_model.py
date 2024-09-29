import re
from typing import Optional

import torch
import transformers
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import sigmoid
from torch.nn import BCELoss
from torchmetrics.classification import MultilabelPrecisionRecallCurve

from src.bert_model import BertForSequenceClassificationWithoutPooling
from src.metrics import create_metrics, create_main_diagnosis_metrics, \
    merge_and_reset_metrics


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

        self.model = BertForSequenceClassificationWithoutPooling.from_pretrained(encoder_model_name,
                                                                                 num_labels=num_classes)
        self.forward = self.model.forward
        self.num_classes = num_classes

        self.pr_curve = MultilabelPrecisionRecallCurve(num_labels=self.num_classes)
        metrics = create_metrics(self.num_classes)
        self.test_metrics = metrics.clone('Test/')
        self.tuned_test_metrics = metrics.clone('Test/Tuned')
        self.val_metrics = metrics.clone('Val/')
        self.tuned_val_metrics = metrics.clone('Val/Tuned')

        main_metrics = create_main_diagnosis_metrics(self.num_classes)
        self.main_test_metrics = main_metrics.clone('Test/')
        self.main_val_metrics = main_metrics.clone('Val/')

        self.val_preds, self.val_labels = [], []
        self.val_loss = BCELoss()

        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr = lr

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

    def training_step(self, batch, batch_idx):
        loss = self(batch['input_ids'], batch['attention_mask'], labels=batch['labels'])[0]
        self.log("Train/Loss", loss)
        return loss

    def test_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        result = self(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], return_dict=True, tuned=True)
        self.log("Test/Loss", result['untuned_loss'])
        self.log("Test/TunedLoss", result['loss'])

        preds = sigmoid(result['untuned_logits'])
        tuned_preds = sigmoid(result['logits'])
        targets = batch['labels'].long()

        self.test_metrics.update(preds, targets, indexes=batch['query_idces'])
        self.tuned_test_metrics.update(tuned_preds, targets, indexes=batch['query_idces'])
        self.main_test_metrics.update(preds, batch['first_codes'])

        self.test_preds.append(preds)
        self.test_labels.append(targets)

        return result['loss']

    def validation_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        result = self(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], return_dict=True)

        self.log("Val/Loss", result['loss'], on_step=True, on_epoch=True)

        preds = sigmoid(result['logits'])
        targets = batch['labels'].long()

        self.pr_curve.update(preds, targets)
        self.val_metrics.update(preds, targets, indexes=batch['query_idces'])
        self.main_val_metrics.update(preds, batch['first_codes'])

        self.val_preds.append(preds)
        self.val_labels.append(targets)

        return result['loss']

    def on_test_epoch_end(self) -> None:
        self.log_dict(merge_and_reset_metrics(self.test_metrics, self.main_test_metrics, self.tuned_test_metrics))

    def on_validation_epoch_end(self) -> None:
        tensor_preds = torch.concat(self.val_preds)
        tensor_labels = torch.concat(self.val_labels)

        self.model.tune_thresholds(tensor_preds, tensor_labels)
        scaled_preds = tensor_preds * 0.5 / self.model.thresholds

        indexes = torch.arange(len(scaled_preds)).unsqueeze(1).expand(len(scaled_preds), self.num_classes)
        self.tuned_val_metrics.update(scaled_preds, tensor_labels, indexes=indexes)

        metrics = merge_and_reset_metrics(self.val_metrics, self.main_val_metrics, self.tuned_val_metrics)
        metrics |= {'Val/TunedLoss_epoch': self.val_loss(tensor_preds, tensor_labels)}
        self.log_dict(metrics)

        self.val_preds.clear()
        self.val_labels.clear()

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
