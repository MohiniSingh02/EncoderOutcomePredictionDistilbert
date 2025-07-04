import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, BertConfig

from src.model.bert_model import DistilBertForSequenceClassification
from src.model.dataset import MIMICClassificationDataModule, extract_re_group, preprocess, load_data_from, \
    ClassificationCollator
from src.model.lightning_model import ClassificationModel

torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description="Script for processing and training with configurable parameters.")

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("../../../data"),
        help="Path to the data directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="kamalkraj/distilBioBERT",
        help="Pretrained model identifier or path"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading"
    )
    parser.add_argument(
        "--_truncate_again",
        type=bool,
        default=True,
        help="Whether to apply truncation again"
    )

    return parser.parse_args()

def load_data_split():


 if __name__ == "__main__":
    args = parse_args()

    data_dir = args.data_dir.absolute()

    icd_version = int(extract_re_group(str(data_dir), r'icd-?(\d{1,2})'))
    hospital_type = extract_re_group(str(data_dir), r'(icu|hosp)')

    split = 'test'

    df = preprocess(load_data_from(data_dir, f'*{split}*'), icd_version)


    model = DistilBertForSequenceClassification.from_pretrained(args.pretrained_model).cuda()
    ClassificationCollator(model.config)

    tuned_results = []
    untuned_results = []
    labels = []

    for batch in tqdm(DataLoader(df,
                          batch_size=args.eval_batch_size,
                          collate_fn=args.collator,
                          pin_memory=True,
                          num_workers=args.num_workers,
                          persistent_workers=args.num_workers > 0)):
        batch_result = model(batch['input_ids'].cuda(), batch['attention_mask'].cuda(), return_dict=True, tuned=True)
        tuned_results.extend(model.get_labels_from_result(batch_result.logits))
        untuned_results.extend(model.get_labels_from_result(batch_result.untuned_logits))
        labels.extend(batch['raw_labels'])

    with open('tuned_results.json', 'w') as f:
        json.dump({
            'predicted_codes': tuned_results,
            'annotated_codes': labels,
            'unique_labels': set([label for result in tuned_results for label in result])
        }, f)
    with open('untuned_results.json', 'w') as f:
        json.dump({
            'predicted_codes': untuned_results,
            'annotated_codes': labels,
            'unique_labels': set([label for result in untuned_results for label in result])
        }, f)
