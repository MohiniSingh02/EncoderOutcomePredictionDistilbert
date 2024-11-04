from typing import Optional, Union, Tuple

import torch
from dataclasses import dataclass
from torch import tensor
from torch.nn import BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertConfig, BertModel, MegatronBertModel
from transformers.utils import ModelOutput


@dataclass
class TunedSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    untuned_loss: Optional[torch.FloatTensor] = None
    untuned_logits: Optional[torch.FloatTensor] = None


class BertForSequenceClassificationWithoutPooling(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config

        if 'megatron' in config.model_type:
            config._attn_implementation = 'eager'
            self.bert = MegatronBertModel(config, add_pooling_layer=False)
        else:
            self.bert = BertModel(config, add_pooling_layer=False)

        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.register_buffer('tuning_weights', torch.empty(config.num_labels, dtype=torch.float))
        self.register_buffer('thresholds', torch.empty(config.num_labels, dtype=torch.float))

        # hardcode these for our case
        self.loss = BCEWithLogitsLoss()
        self.config.problem_type = "multi_label_classification"

        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            tuned: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TunedSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_state_cls_token = outputs[0][:, 0]

        logits = self.classifier(last_state_cls_token)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        untuned_logits = logits
        untuned_loss = loss
        if tuned:
            logits = self.tuning_weights + logits
            if labels is not None:
                loss = self.loss(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TunedSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            untuned_loss=untuned_loss,
            untuned_logits=untuned_logits,
        )

    def tune_thresholds(self, preds: tensor, labels: tensor) -> (tensor, dict[str, tensor]):
        from torchmetrics.functional.classification import multilabel_precision_recall_curve

        pr_curve_results = multilabel_precision_recall_curve(preds, labels, self.config.num_labels)
        for i, (p, r, t) in enumerate(zip(*pr_curve_results)):
            # remove last entry which is just there for backwards compatibility
            f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1])
            f1 = torch.nan_to_num(f1, 0)
            max_f1, ix = torch.max(f1, dim=0)

            if max_f1 == 0:  # if there's no successful threshold, use  at least 0.5 or the biggest and then some
                self.thresholds[i] = max(t[-1] + 1e-6, 0.5)
            elif ix == 0:  # if the first is the best use something a little lower or 0.5 if it's smaller
                self.thresholds[i] = min(t[0] - 1e-6, 0.5)
            else:  # else take the middle between the best and the previous one
                self.thresholds[i] = (t[ix - 1] + t[ix]) / 2

            self.tuning_weights = torch.log(1 / self.thresholds - 1)

    def get_labels_from_result(self, logits: tensor, topk: int = None):
        if topk is None:
            discretized_batch = logits > 0
            indices_batch = [discretized.nonzero(as_tuple=True)[0] for discretized in discretized_batch]
        else:
            indices_batch = logits.topk(topk).indices

        return [[self.config.id2label[index.item()] for index in indices] for indices in indices_batch]
