import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from pandas import DataFrame
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig


class MIMICClassificationDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: Path = Path("../../../data"),
                 batch_size: int = 4,
                 eval_batch_size: int = 4,
                 pretrained_model: str = "kamalkraj/distilBioBERT",
                 num_workers: int = 0,
                 truncate_again: bool = False):
        super().__init__()
        self.icd_version = int(extract_re_group(str(data_dir), r'icd-?(\d{1,2})'))
        self.hospital_type = extract_re_group(str(data_dir), r'(icu|hosp)')
        self.save_hyperparameters("data_dir", "batch_size", "eval_batch_size", "pretrained_model", "num_workers", "truncate_again")
        self.data_dir = data_dir.absolute()
        self.truncate_again = truncate_again

        print(f"[DataModule] Truncate again: {self.truncate_again}")
        print(f"[DEBUG] Using data_dir: {self.data_dir}")
        print(f"[DEBUG] Truncate again = {truncate_again}")

        # Load and cache splits
        splits_icd9 = load_splits(data_dir, 9, truncate_labels=self.truncate_again)
        splits_icd10 = load_splits(data_dir, 10, truncate_labels=self.truncate_again)

        self.label2id = {'icd9': compute_label_idx(*splits_icd9.values()),
                 'icd10': compute_label_idx(*splits_icd10.values())}
        
        self.num_labels = {
        'icd9': len(self.label2id['icd9']),
        'icd10': len(self.label2id['icd10'])
         }
        # Combine data into a unified structure with task_name
        self.data = {'train': [], 'val': [], 'test': []}
        for version, splits in [('icd9', splits_icd9), ('icd10', splits_icd10)]:
            for split_name, df in splits.items():
                df['task_name'] = version
                self.data[split_name].extend(df.to_dict('records'))

        # Tokenizer and collator
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.collator = ClassificationCollator(self.config)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.mimic_train, self.mimic_val, self.mimic_test = None, None, None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.mimic_train = ClassificationDataset(self.data['train'], self.label2id)
            self.mimic_val = ClassificationDataset(self.data['val'], self.label2id)
            print("Train Length: ", len(self.mimic_train))
            print("Val length: ", len(self.mimic_val))
        if stage == "test" or stage is None:
            self.mimic_test = ClassificationDataset(self.data['test'], self.label2id)
            print("Test length: ", len(self.mimic_test))

    def train_dataloader(self):
        return DataLoader(self.mimic_train,
                          batch_size=self.batch_size,
                          collate_fn=self.collator,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          shuffle=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.mimic_val,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.collator,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.mimic_test,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.collator,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)


def extract_re_group(input_string, pattern):
    match = re.search(pattern, input_string)
    return match.group(1) if match else 'not found'


def load_splits(data_dir, icd_version, truncate_labels=True):
    return {
        split: preprocess(load_data_from(data_dir, f'*{split}*'), icd_version if truncate_labels else None)
        for split in ['train', 'val', 'test']
    }


def load_data_from(path: Path, glob: str):
    files = list(path.glob(glob))
    logging.warning(f'Detected {len(files)} files for glob {glob}: {files}')
    dfs = []

    for file in files:
        try:
            if file.suffix == '.csv':
                df = pd.read_csv(file)
            elif file.suffix == '.parquet':
                df = pd.read_parquet(file)
            else:
                logging.warning(f"Unsupported file type: {file.suffix}")
                continue

            df.columns = df.columns.str.lower()
            if 'test' in df.columns:
                df.drop(columns='test', inplace=True)
            if 'short_codes' in df and 'labels' not in df.columns:
                df.rename(columns={'short_codes': 'labels'}, inplace=True)
            if not df.empty  and 'labels' in df.columns and isinstance(df.labels.iloc[0], str):
                df.labels = df.labels.str.replace(r"[\[\]' ]", "", regex=True).str.split(",")
            if 'text' in df.columns:
                    df.text = df.text.astype(str).str.strip()

            dfs.append(df)
        except Exception as e:
            logging.error(f"Failed to process file {file.name}: {e}")

    if not dfs:
        raise ValueError(f"No valid dataframes could be loaded from {path} with glob '{glob}'")

    return pd.concat(dfs, ignore_index=True)


def preprocess(df: DataFrame, truncate_to_icd_version: int) -> DataFrame:
    if truncate_to_icd_version:
        print(f"[INFO] Truncating to ICD-{truncate_to_icd_version}")
    else:
        print("[INFO] Skipping truncation")
    
    filtered = df[df.labels.str.len().astype(bool)].copy()
    if truncate_to_icd_version is not None:
        filtered['labels'] = (
            filtered['labels'].apply(truncate_labels_9 if truncate_to_icd_version == 9 else truncate_labels_10)
        )
    return filtered


#def filter_empty_labels(df: DataFrame) -> DataFrame:
   # return df[df.labels.str.len().astype(bool)]


def truncate_labels_9(labels: list[str]) -> list[str]:
    clean = []
    for label in labels:
        if not isinstance(label, str):
            continue
        label = label.strip()
        if not label:
            continue
        clean.append(label[:4] if label.startswith('E') else label[:3])
    return clean

def truncate_labels_10(labels: list[str]) -> list[str]:
    return [label[:3] for label in labels if isinstance(label, str) and label.strip()]


def compute_label_idx(*dfs: DataFrame) -> dict[str, int]:
    labels: pd.Series = pd.concat([df['labels'] for df in dfs], ignore_index=True)
    label_distribution = labels.explode().value_counts()
    label_idx = {label: idx for idx, label in enumerate(label_distribution.index)}
    return label_idx


class ClassificationCollator:
    def __init__(self, config: PretrainedConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path, config=config,
                                                       model_max_length=config.max_position_embeddings)

    def __call__(self, data):
        admission_notes = [x['admission_note'] for x in data]
        tokenized = self.tokenizer(admission_notes, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        labels = torch.stack([x['labels'] for x in data])
        first_codes = torch.tensor([x['first_code'] for x in data])
        query_idces = first_codes.unsqueeze(1).expand_as(labels).contiguous()

        return {"input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels,
                "lengths": torch.sum(tokenized["attention_mask"], dim=1),
                'query_idces': query_idces,
                'first_codes': first_codes,
                'raw_labels': [x['raw_labels'] for x in data],
                'task_names': [x['task_name'] for x in data]
                }


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, label2id):
        # tokenize admission notes
        self.examples = examples
        self.label2id = label2id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        note = example['text']
        labels = example['labels']
        task = example['task_name']
        hadm_id = example['hadm_id']
        
        label_map = self.label2id[task]
        label_ids = [label_map[x] for x in labels if x in label_map]
        label_tensor = torch.zeros(len(label_map), dtype=torch.float32)
        if label_ids:
            label_tensor[label_ids] = 1

        return {'admission_note': note,
                'labels': label_tensor,
                'hadm_id': hadm_id,
                'first_code': label_ids[0] if label_ids else -1,
                'idx': idx,
                'raw_labels': labels,
                'task_name': task
                }
