from typing import Optional

import torch
import transformers
from transformers import AutoModel, AutoConfig
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import sigmoid
from torch.nn import BCELoss
from torchmetrics.classification import MultilabelPrecisionRecallCurve

from src.model.bert_model import DistilBertForSequenceClassification
from src.model.dataset import MIMICClassificationDataModule
from src.model.metrics import create_metrics, create_main_diagnosis_metrics, merge_and_reset_metrics


class ClassificationModel(LightningModule):
    def __init__(self,
                 num_classes: int = 1446,
                 warmup_steps: int = 0,
                 decay_steps: int = 50_000,
                 weight_decay: float = 0.01,
                 lr: float = 2e-5,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = None
        self.config = None
        self.num_classes_icd9 = None
        self.num_classes_icd10 = None

        self.val_preds = []
        self.val_labels = []
        self.val_loss = BCELoss()

        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.weight_decay = weight_decay
        self.lr = lr

        # Metrics
        self.pr_curve = MultilabelPrecisionRecallCurve(num_labels=self.num_classes)
        metrics = create_metrics(self.num_classes)
        self.test_metrics = metrics.clone('Test/')
        self.tuned_test_metrics = metrics.clone('Test/Tuned')
        self.val_metrics = metrics.clone('Val/')
        self.tuned_val_metrics = metrics.clone('Val/Tuned')

        main_metrics = create_main_diagnosis_metrics(self.num_classes)
        self.main_test_metrics = main_metrics.clone('Test/')
        self.main_val_metrics = main_metrics.clone('Val/')

    def setup(self, stage: Optional[str] = None):
        if self.trainer:
           for cb in self.trainer.callback_connector.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.filename = 'lastckpt_' + cb.filename

            if self.trainer.datamodule is not None:
                data_module: MIMICClassificationDataModule = self.trainer.datamodule
                self.save_hyperparameters({
                    'icd': data_module.icd_version,
                    'split': data_module.hospital_type,
                    'encoder_model_name': data_module.config.name_or_path
                })
                self.config = AutoConfig.from_pretrained("kamalkraj/distilBioBERT")
                self.encoder = AutoModel.from_pretrained("kamalkraj/distilBioBERT", config=self.config)
                # Figure out number of classes for each task
                self.num_classes_icd9 = data_module.num_labels['icd9']
                self.num_classes_icd10 = data_module.num_labels['icd10']
                hidden_size = self.config.hidden_size

                # Define task-specific classification heads
                self.classifier_icd9 = torch.nn.Linear(hidden_size, self.num_classes_icd9)
                self.classifier_icd10 = torch.nn.Linear(hidden_size, self.num_classes_icd10)

                #Update global class count for shared metrics
                self.num_classes = max(self.num_classes_icd9, self.num_classes_icd10)
                self.pr_curve = MultilabelPrecisionRecallCurve(num_labels=self.num_classes)
                metrics = create_metrics(self.num_classes)
                self.test_metrics = metrics.clone('Test/')
                self.tuned_test_metrics = metrics.clone('Test/Tuned')
                self.val_metrics = metrics.clone('Val/')
                self.tuned_val_metrics = metrics.clone('Val/Tuned')
                main_metrics = create_main_diagnosis_metrics(self.num_classes)
                self.main_test_metrics = main_metrics.clone('Test/')
                self.main_val_metrics = main_metrics.clone('Val/')

    
    def forward(self, input_ids, attention_mask, labels=None, task=None, return_dict=False, tuned=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token

        if task == 'icd9':
            logits = self.classifier_icd9(pooled_output)
        elif task == 'icd10':
            logits = self.classifier_icd10(pooled_output)
        else:
            raise ValueError("Task must be 'icd9' or 'icd10'")

        loss = None
        if labels is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())

        if return_dict:
            return {'loss': loss, 'logits': logits, 'untuned_loss': loss, 'untuned_logits': logits}
        else:
            return (loss, logits)
    
    
    def training_step(self, batch, batch_idx):
        task = batch['task']
        loss, _ = self(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], task=task)
        self.log("Train/Loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        return self(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], task=batch['task'], return_dict=True, tuned=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        result = self(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], task=batch['task'], return_dict=True, tuned=True)
        self.log("Test/Loss", result['untuned_loss'])
        self.log("Test/TunedLoss", result['loss'])

        preds = sigmoid(result['untuned_logits'])
        tuned_preds = sigmoid(result['logits'])
        targets = batch['labels'].long()

        self.test_metrics.update(preds, targets, indexes=batch['query_idces'])
        self.tuned_test_metrics.update(tuned_preds, targets, indexes=batch['query_idces'])
        self.main_test_metrics.update(preds, batch['first_codes'])

        return result['loss']

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        result = self(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], task=batch['task'], return_dict=True)

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

        #self.model.tune_thresholds(tensor_preds, tensor_labels)
        #scaled_preds = tensor_preds * 0.5 / self.model.thresholds
        #basic tuning logic 
        scaled_preds = tensor_preds 
        indexes = torch.arange(len(scaled_preds), device=self.device).unsqueeze(1).expand(len(scaled_preds),
                                                                                          self.num_classes)
        self.tuned_val_metrics.update(scaled_preds, tensor_labels, indexes=indexes)

        metrics = merge_and_reset_metrics(self.val_metrics, self.main_val_metrics, self.tuned_val_metrics)
        metrics |= {'Val/TunedLoss_epoch': self.val_loss(tensor_preds, tensor_labels.to(tensor_preds.dtype))}
        self.log_dict(metrics)

        self.val_preds.clear()
        self.val_labels.clear()

    def configure_optimizers(self, param_optimizer=None):
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

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
