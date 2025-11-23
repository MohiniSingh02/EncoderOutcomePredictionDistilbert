from typing import Optional

import torch
import transformers
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import sigmoid
from torch.nn import BCELoss, Linear, BCEWithLogitsLoss
from torchmetrics.classification import MultilabelPrecisionRecallCurve

from src.model.bert_model import BertForSequenceClassificationWithoutPooling
from src.model.dataset import MIMICClassificationDataModule
from src.model.metrics import create_metrics, create_main_diagnosis_metrics, merge_and_reset_metrics


class ClassificationModel(LightningModule):
    def __init__(self,
                 multitask: bool = False,
                 num_classes: int = 1446,
                 warmup_steps: int = 0,
                 decay_steps: int = 50_000,
                 weight_decay: float = 0.01,
                 lr: float = 2e-5,
                 num_classes_icd9: Optional[int] = None,
                 num_classes_icd10: Optional[int] = None,
                 ):
        super().__init__()
        self.tuned_test_metrics = None
        self.val_labels = None
        self.val_preds = None
        self.main_val_metrics = None
        self.main_test_metrics = None
        self.tuned_val_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        self.val_labels_icd10 = None
        self.val_preds_icd10 = None
        self.val_labels_icd9 = None
        self.val_preds_icd9 = None
        self.test_metrics_icd10 = None
        self.val_metrics_icd10 = None
        self.pr_curve_icd10 = None
        self.val_metrics_icd9 = None
        self.test_metrics_icd9 = None
        self.pr_curve_icd9 = None
        self.pr_curve = None
        self.save_hyperparameters(ignore=["model"])

        self.model = None
        self.multitask = multitask

        self.num_classes_icd9 = num_classes_icd9
        self.num_classes_icd10 = num_classes_icd10
        self.num_classes = num_classes

        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.weight_decay = weight_decay
        self.lr = lr

        self.icd9_head: Optional[Linear] = None
        self.icd10_head: Optional[Linear] = None
        self.val_loss = BCELoss()

    def setup(self, stage: Optional[str] = None):
        if self.trainer is None:
            raise NotImplementedError("Usage without Trainer and DataModule isn't currently intended.")

        checkpoint_callback = None
        # getting the first checkpoint callback if it exists
        if hasattr(self.trainer, "checkpoint_callbacks") and self.trainer.checkpoint_callbacks:
            checkpoint_callback = self.trainer.checkpoint_callbacks[0]
        if checkpoint_callback is not None:
            # Handle both old and new versions of checkpoint callback attributes
            filename_attr = getattr(checkpoint_callback, "filename", None)
            if filename_attr is not None:
                checkpoint_callback.CHECKPOINT_NAME_LAST = f"lastckpt_{filename_attr}"
            else:
                checkpoint_callback.CHECKPOINT_NAME_LAST = 'lastckpt'

        data_module: MIMICClassificationDataModule = self.trainer.datamodule
        self.save_hyperparameters({
            "icd": getattr(data_module, "icd_version", "unknown"),
            "split": getattr(data_module, "hospital_type", "unknown"),
            "encoder_model_name": data_module.config.name_or_path,
        })

        # load encoder
        self.model = BertForSequenceClassificationWithoutPooling.from_pretrained(
            data_module.config.name_or_path, config=data_module.config
        )
        hidden_size = getattr(self.model.config, "hidden_size", 768)

        # Metrics setup
        if self.multitask:
            # derive class counts from datamodule
            self.num_classes_icd9 = getattr(data_module, "num_labels_icd9")
            self.num_classes_icd10 = getattr(data_module, "num_labels_icd10")

            # classification heads
            self.icd9_head = Linear(hidden_size, self.num_classes_icd9)
            self.icd10_head = Linear(hidden_size, self.num_classes_icd10)

            # ICD9 metrics
            self.pr_curve_icd9 = MultilabelPrecisionRecallCurve(num_labels=self.num_classes_icd9)
            metrics_icd9 = create_metrics(self.num_classes_icd9)
            self.test_metrics_icd9 = metrics_icd9.clone('Test/ICD9/')
            self.val_metrics_icd9 = metrics_icd9.clone('Val/ICD9/')

            # ICD10 metrics
            self.pr_curve_icd10 = MultilabelPrecisionRecallCurve(num_labels=self.num_classes_icd10)
            metrics_icd10 = create_metrics(self.num_classes_icd10)
            self.val_metrics_icd10 = metrics_icd10.clone("Val/ICD10/")
            self.test_metrics_icd10 = metrics_icd10.clone("Test/ICD10/")

            self.val_preds_icd9, self.val_labels_icd9 = [], []
            self.val_preds_icd10, self.val_labels_icd10 = [], []

            print(
                f"[ClassificationModel.setup] MULTITASK=True | "
                f"num_classes_icd9={self.num_classes_icd9}, num_classes_icd10={self.num_classes_icd10}"
            )

        # single task setup
        else:
            self.num_classes = getattr(data_module, "num_labels", self.num_classes)

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

            print(f"[ClassificationModel.setup] MULTITASK=False | num_classes={self.num_classes}")

    def forward(self, input_ids, attention_mask, task: Optional[str] = None, labels=None,
                return_dict: bool = True, **kwargs,):

        # MULTITASK
        if self.multitask:
            outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = outputs[0][:, 0, :]  # [CLS] token

            logits_icd9 = self.icd9_head(pooled)
            logits_icd10 = self.icd10_head(pooled)

            loss = None
            if task is not None and labels is not None:
                if task == "icd9":
                    loss = BCEWithLogitsLoss()(logits_icd9, labels)
                elif task == "icd10":
                    loss = BCEWithLogitsLoss()(logits_icd10, labels)
                else:
                    raise ValueError(f"Unknown task: {task}")

            return {
                "loss": loss,
                "logits_icd9": logits_icd9,
                "logits_icd10": logits_icd10,
            }

        # SINGLE TASK
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs,
        )

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        if self.multitask:

            # handle CombinedLoader(zip)
            if isinstance(batch, dict) and "icd9" in batch and "icd10" in batch:
                # Interleaved multitask batches (icd9 + icd10 together)
                total_loss = 0.0

                for task_name in ["icd9", "icd10"]:
                    b = batch[task_name]

                    result = self(
                        input_ids=b["input_ids"],
                        attention_mask=b["attention_mask"],
                        labels=b["labels"],
                        task=task_name,
                    )

                    loss = result["loss"]
                    total_loss += loss

                    self.log(
                        f"Train/Loss_{task_name}",
                        loss,
                        prog_bar=True,
                        on_epoch=True,
                        batch_size=b["input_ids"].size(0),
                    )

                return total_loss

            # If not using CombinedLoader
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            task = self._infer_task(batch)

            result = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                task=task,
            )

            loss = result["loss"]
            self.log(
                f"Train/Loss_{task}",
                loss,
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["input_ids"].size(0),
            )
            return loss

        else:
            # Single-task unchanged
            result = self(
                batch["input_ids"], batch["attention_mask"], labels=batch["labels"]
            )
            loss = (
                result[0] if isinstance(result, (list, tuple)) else result["loss"]
            )
            self.log("Train/Loss", loss)
            return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.multitask:
            task = self._infer_task(batch)
            return self(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        task=task,
                        return_dict=True)
        else:
            return self(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], return_dict=True, tuned=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Optional[STEP_OUTPUT]:
        if self.multitask:
            task = self._infer_task(batch)

            result = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                          labels=batch["labels"], task=task)

            preds = sigmoid(result[f"logits_{task}"])
            targets = batch["labels"].long()

            if task == "icd9":
                self.test_metrics_icd9.update(preds, targets, indexes=batch["query_idces"])
            else:
                self.test_metrics_icd10.update(preds, targets, indexes=batch["query_idces"])
            # Log test loss with batch size
            self.log(
                f"Test/Loss_{task}",
                result["loss"],
                on_step=False,
                on_epoch=True,
                batch_size=batch["input_ids"].size(0)
            )
            return result["loss"]

        else:
            result = self(batch['input_ids'], batch['attention_mask'], labels=batch['labels'],
                          return_dict=True, tuned=True)
            self.log("Test/Loss", result['untuned_loss'], on_epoch=True, batch_size=batch["input_ids"].size(0))
            self.log("Test/TunedLoss", result['loss'], on_epoch=True, batch_size=batch["input_ids"].size(0))

            preds = sigmoid(result['untuned_logits'])
            tuned_preds = sigmoid(result['logits'])
            targets = batch['labels'].long()

            self.test_metrics.update(preds, targets, indexes=batch['query_idces'])
            self.tuned_test_metrics.update(tuned_preds, targets, indexes=batch['query_idces'])
            self.main_test_metrics.update(preds, batch['first_codes'])
            return result['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Optional[STEP_OUTPUT]:
        if self.multitask:
            num_labels = batch["labels"].shape[1]

            if num_labels == self.num_classes_icd9:
                task = "icd9"
            elif num_labels == self.num_classes_icd10:
                task = "icd10"
            else:
                raise ValueError(f"Unknown label dimension: {num_labels}")

            result = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                          labels=batch["labels"], task=task)

            preds = sigmoid(result[f"logits_{task}"])
            targets = batch["labels"].long()
            self.log(f"Val/Loss_{task}", result["loss"], on_step=True, on_epoch=True)

            if task == "icd9":
                self.pr_curve_icd9.update(preds, targets)
                self.val_metrics_icd9.update(preds, targets, indexes=batch["query_idces"])
                self.val_preds_icd9.append(preds)
                self.val_labels_icd9.append(targets)
            else:
                self.pr_curve_icd10.update(preds, targets)
                self.val_metrics_icd10.update(preds, targets, indexes=batch["query_idces"])
                self.val_preds_icd10.append(preds)
                self.val_labels_icd10.append(targets)
            return result["loss"]
        else:
            result = self(batch['input_ids'], batch['attention_mask'], labels=batch['labels'], return_dict=True)
            self.log("Val/Loss", result['loss'], on_step=True, on_epoch=True, batch_size=batch["input_ids"].size(0))

            preds = sigmoid(result['logits'])
            targets = batch['labels'].long()
            self.pr_curve.update(preds, targets)
            self.val_metrics.update(preds, targets, indexes=batch['query_idces'])
            self.main_val_metrics.update(preds, batch['first_codes'])
            self.val_preds.append(preds)
            self.val_labels.append(targets)
            return result['loss']

    def on_test_epoch_end(self) -> None:
        if self.multitask:
            metrics_icd9 = merge_and_reset_metrics(self.test_metrics_icd9)
            metrics_icd10 = merge_and_reset_metrics(self.test_metrics_icd10)
            self.log_dict(metrics_icd9)
            self.log_dict(metrics_icd10)
        else:
            self.log_dict(merge_and_reset_metrics(self.test_metrics, self.main_test_metrics, self.tuned_test_metrics))

    def on_validation_epoch_end(self) -> None:
        if self.multitask:
            # Only compute harmonic mean if both metrics exist
            metrics_icd9, metrics_icd10 = {}, {}

            # ICD9
            if self.val_preds_icd9:
                tensor_preds = torch.concat(self.val_preds_icd9)
                tensor_labels = torch.concat(self.val_labels_icd9)
                metrics_icd9 = merge_and_reset_metrics(self.val_metrics_icd9)
                metrics_icd9 |= {
                    "Val/ICD9/Loss_epoch": self.val_loss(tensor_preds, tensor_labels.to(tensor_preds.dtype))}
                self.log_dict(metrics_icd9)
                self.val_preds_icd9.clear()
                self.val_labels_icd9.clear()

            # ICD10
            if self.val_preds_icd10:
                tensor_preds = torch.concat(self.val_preds_icd10)
                tensor_labels = torch.concat(self.val_labels_icd10)
                metrics_icd10 = merge_and_reset_metrics(self.val_metrics_icd10)
                metrics_icd10 |= {
                    "Val/ICD10/Loss_epoch": self.val_loss(tensor_preds, tensor_labels.to(tensor_preds.dtype))}
                self.log_dict(metrics_icd10)
                self.val_preds_icd10.clear()
                self.val_labels_icd10.clear()

            # Compute and log harmonic mean AUROC across ICD9 & ICD10
            if "Val/ICD9/AUROC" in metrics_icd9 and "Val/ICD10/AUROC" in metrics_icd10:
                icd9_auroc = metrics_icd9["Val/ICD9/AUROC"]
                icd10_auroc = metrics_icd10["Val/ICD10/AUROC"]

                harmonic_mean = 2 * (icd9_auroc * icd10_auroc) / (icd9_auroc + icd10_auroc + 1e-8)
                arithmetic_mean = (icd9_auroc + icd10_auroc) / 2

                self.log("Val/HARMONIC/AUROC", harmonic_mean, prog_bar=True, sync_dist=True)
                self.log("Val/AVG/AUROC", arithmetic_mean, prog_bar=False, sync_dist=True)
        else:
            tensor_preds = torch.concat(self.val_preds)
            tensor_labels = torch.concat(self.val_labels)

            # Threshold tuning for single-task
            self.model.tune_thresholds(tensor_preds, tensor_labels)
            scaled_preds = tensor_preds * 0.5 / self.model.thresholds
            indexes = torch.arange(len(scaled_preds), device=self.device).unsqueeze(1).expand(len(scaled_preds),
                                                                                              self.num_classes)
            self.tuned_val_metrics.update(scaled_preds, tensor_labels, indexes=indexes)
            metrics = merge_and_reset_metrics(self.val_metrics, self.main_val_metrics, self.tuned_val_metrics)
            metrics |= {'Val/TunedLoss_epoch': self.val_loss(tensor_preds, tensor_labels.to(tensor_preds.dtype))}
            self.log_dict(metrics)
            self.val_preds.clear()
            self.val_labels.clear()

    def _infer_task(self, batch) -> str:
        num_labels = batch["labels"].shape[1]
        if num_labels == self.num_classes_icd9:
            return "icd9"
        if num_labels == self.num_classes_icd10:
            return "icd10"
        raise ValueError(f"Unknown label dimension: {num_labels}")

    def configure_optimizers(self):
        param_optimizer = [(n, p) for n, p in self.named_parameters() if 'pooler' not in n]
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

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
