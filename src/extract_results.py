import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.bert_model import BertForSequenceClassificationWithoutPooling
from src.model.dataset import (
    load_data_from,
    preprocess,
    extract_re_group,
    ClassificationCollator
)

torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description="Script for processing and training with configurable parameters.")

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("../../../data"),
        help="Path to single-task data directory (e.g., icd9/hosp)."
    )
    parser.add_argument(
        "--multitask",
        action="store_true",
        help="Enable multitask mode (uses ICD9 and ICD10 data)."
    )
    parser.add_argument(
        "--icd9_data_dir",
        type=Path,
        default=None,
        help="ICD9 data directory if multitask."
    )
    parser.add_argument(
        "--icd10_data_dir",
        type=Path,
        default=None,
        help="ICD10 data directory if multitask."
    )
    # Model/runtime
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        help="Pretrained model identifier or path."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading"
    )
    parser.add_argument(
        "--truncate_again",
        type=bool,
        default=True,
        help="Whether to apply truncation again"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("."),
        help="Directory to write results JSONs.",
    )

    return parser.parse_args()


def run_inference(data_dir: Path, model, collator, args, prefix: str):
    """Runs inference for one dataset (ICD9 or ICD10)."""
    icd_version = int(extract_re_group(str(data_dir), r"icd-?(\d{1,2})"))
    split = "test"

    df = preprocess(load_data_from(data_dir, f"*{split}*"), icd_version)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(df["input_ids"].tolist()),
        torch.tensor(df["attention_mask"].tolist()),
        torch.tensor(df["labels"].tolist()),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    tuned_results, untuned_results, labels = [], [], []

    for batch in tqdm(loader, desc=f"Evaluating {prefix} ({icd_version})"):
        batch = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
        batch_result = model(**batch, return_dict=True, tuned=True)

        tuned_results.extend(model.get_labels_from_result(batch_result.logits))
        untuned_results.extend(model.get_labels_from_result(batch_result.untuned_logits))
        labels.extend(batch["raw_labels"])

    results = {
        "predicted_codes_tuned": tuned_results,
        "predicted_codes_untuned": untuned_results,
        "annotated_codes": labels,
        "unique_labels": list({label for result in tuned_results for label in result}),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"results_{prefix}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BertForSequenceClassificationWithoutPooling.from_pretrained(args.pretrained_model).to(device)
    collator = ClassificationCollator(model.config)

    if args.multitask:
        if not (args.icd9_data_dir and args.icd10_data_dir):
            raise ValueError("--multitask requires both --icd9_data_dir and --icd10_data_dir.")
        run_inference(args.icd9_data_dir, model, collator, args, prefix="icd9")
        run_inference(args.icd10_data_dir, model, collator, args, prefix="icd10")
    else:
        if not args.data_dir:
            raise ValueError("Provide --data_dir for single-task mode.")
        run_inference(args.data_dir, model, collator, args, prefix="single")
