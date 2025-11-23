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
from lightning.pytorch.utilities.combined_loader import CombinedLoader


class MIMICClassificationDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: Path = Path("../../../data"),
                 icd9_data_dir: Optional[Path] = None,
                 icd10_data_dir: Optional[Path] = None,
                 multitask: bool = False,
                 batch_size: int = 4,
                 eval_batch_size: int = 4,
                 pretrained_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 truncate_again: bool = True):
        super().__init__()
        self.multitask = multitask
        # self.icd_version = int(extract_re_group(str(data_dir), r'icd-?(\d{1,2})'))
        # self.hospital_type = extract_re_group(str(data_dir), r'(icu|hosp)')
        self.save_hyperparameters()

        if multitask:
            assert icd9_data_dir is not None and icd10_data_dir is not None, \
                "For multitask mode, you must set both icd9_data_dir and icd10_data_dir"

            self.icd9_data_dir = Path(icd9_data_dir).absolute()
            self.icd10_data_dir = Path(icd10_data_dir).absolute()
            self.icd_version = "multitask"

            # load splits for ICD9 and ICD10
            data_dfs_9 = load_splits(self.icd9_data_dir, 9 if truncate_again else None)
            data_dfs_10 = load_splits(self.icd10_data_dir, 10 if truncate_again else None)

            self.label2id_icd9 = compute_label_idx(*data_dfs_9.values())
            self.label2id_icd10 = compute_label_idx(*data_dfs_10.values())

            self.data_icd9 = {split: split_df.to_dict('records') for split, split_df in data_dfs_9.items()}
            self.data_icd10 = {split: split_df.to_dict('records') for split, split_df in data_dfs_10.items()}

            self.num_labels_icd9 = len(self.label2id_icd9)
            self.num_labels_icd10 = len(self.label2id_icd10)

            self.num_labels = max(self.num_labels_icd9, self.num_labels_icd10)

            self.config = AutoConfig.from_pretrained(pretrained_model)

        else:
            self.data_dir = Path(data_dir).absolute()
            self.icd_version = int(extract_re_group(str(data_dir), r'icd-?(\d{1,2})'))
            self.hospital_type = extract_re_group(str(data_dir), r'(icu|hosp)')

            data_dfs = load_splits(self.data_dir, self.icd_version if truncate_again else None)
            self.label2id = compute_label_idx(*data_dfs.values())
            self.data = {split: split_df.to_dict('records') for split, split_df in data_dfs.items()}

            self.num_labels = len(self.label2id)
            self.config = AutoConfig.from_pretrained(pretrained_model, label2id=self.label2id,
                                                     id2label={v: k for k, v in self.label2id.items()},
                                                     num_labels=self.num_labels)

        import os
        self.num_workers = int(os.environ.get("PYTORCH_NUM_WORKERS", 0))
        # default to 0 for safety
        if self.num_workers is None:
            self.num_workers = 0
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.collator = ClassificationCollator(self.config)

        self.mimic_train, self.mimic_val, self.mimic_test = None, None, None

    def setup(self, stage: Optional[str] = None):
        if self.multitask:
            if stage == "fit" or stage is None:
                self.mimic_train = {
                    "icd9": ClassificationDataset(self.data_icd9['train'], label2id=self.label2id_icd9),
                    "icd10": ClassificationDataset(self.data_icd10['train'], label2id=self.label2id_icd10)
                }
                self.mimic_val = {
                    "icd9": ClassificationDataset(self.data_icd9['val'], label2id=self.label2id_icd9),
                    "icd10": ClassificationDataset(self.data_icd10['val'], label2id=self.label2id_icd10)
                }
                print("ICD9 Train:", len(self.mimic_train["icd9"]))
                print("ICD10 Train:", len(self.mimic_train["icd10"]))
            if stage == "test" or stage is None:
                self.mimic_test = {
                    "icd9": ClassificationDataset(self.data_icd9['test'], label2id=self.label2id_icd9),
                    "icd10": ClassificationDataset(self.data_icd10['test'], label2id=self.label2id_icd10)
                }

                print("ICD9 Test:", len(self.mimic_test["icd9"]))
                print("ICD10 Test:", len(self.mimic_test["icd10"]))

            return

        if stage == "fit" or stage is None:
            self.mimic_train = ClassificationDataset(self.data['train'], label2id=self.label2id)
            self.mimic_val = ClassificationDataset(self.data['val'], label2id=self.label2id)
            print("Train Length:", len(self.mimic_train))
            print("Val Length:", len(self.mimic_val))

        if stage == "test" or stage is None:
            self.mimic_test = ClassificationDataset(self.data['test'], label2id=self.label2id)
            print("Test Length:", len(self.mimic_test))

    def train_dataloader(self):
        if self.multitask:
            icd9_loader = DataLoader(self.mimic_train["icd9"], batch_size=self.batch_size,
                                     collate_fn=self.collator, pin_memory=True,
                                     num_workers=self.num_workers
                                     , shuffle=True)
            icd10_loader = DataLoader(self.mimic_train["icd10"], batch_size=self.batch_size,
                                      collate_fn=self.collator, pin_memory=True,
                                      num_workers=self.num_workers
                                      , shuffle=True)
            return CombinedLoader(
                {"icd9": icd9_loader, "icd10": icd10_loader},
                mode="zip"
            )
        return DataLoader(self.mimic_train, batch_size=self.batch_size,
                          collate_fn=self.collator, pin_memory=True,
                          num_workers=self.num_workers
                          , shuffle=True
                          )

    def val_dataloader(self):
        if self.multitask:
            icd9_loader = DataLoader(
                self.mimic_val["icd9"], batch_size=self.eval_batch_size,
                collate_fn=self.collator, pin_memory=True,
                num_workers=self.num_workers
            )
            icd10_loader = DataLoader(
                self.mimic_val["icd10"], batch_size=self.eval_batch_size,
                collate_fn=self.collator, pin_memory=True,
                num_workers=self.num_workers
            )
            return [icd9_loader, icd10_loader]
        return DataLoader(self.mimic_val, batch_size=self.eval_batch_size,
                          collate_fn=self.collator, pin_memory=True,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        if self.multitask:
            icd9_loader = DataLoader(
                self.mimic_test["icd9"], batch_size=self.eval_batch_size,
                collate_fn=self.collator, pin_memory=True,
                num_workers=self.num_workers
            )
            icd10_loader = DataLoader(
                self.mimic_test["icd10"], batch_size=self.eval_batch_size,
                collate_fn=self.collator, pin_memory=True,
                num_workers=self.num_workers
            )
            return [icd9_loader, icd10_loader]
        return DataLoader(self.mimic_test, batch_size=self.eval_batch_size,
                          collate_fn=self.collator, pin_memory=True,
                          num_workers=self.num_workers)


def extract_re_group(input_string, pattern):
    match = re.search(pattern, input_string)
    return match.group(1) if match else 'not found'


def load_splits(data_dir, icd_version):
    return {
        split: preprocess(load_data_from(data_dir, f'*{split}*'), icd_version)
        for split in ['train', 'val', 'test']
    }


def load_data_from(path: Path, glob: str):
    files = list(path.glob(glob))
    logging.warning(f'Detected {len(files)} files for glob {glob}: {files}')
    dfs = [pd.read_csv(file) if file.suffix == '.csv' else
           pd.read_parquet(file) if file.suffix == '.parquet' else None
           for file in files]
    for df in dfs:
        df.columns = df.columns.str.lower()
        if 'test' in df:
            df.drop(columns='test', inplace=True)
        if 'short_codes' in df and 'labels' not in df:
            df.rename(columns={'short_codes': 'labels'}, inplace=True)
        if isinstance(df.labels.iloc[0], str):
            df.labels = df.labels.str.replace(r"[\[\]' ]", "", regex=True).str.split(",")
        df.text = df.text.str.strip()
    return pd.concat(dfs)


def preprocess(df: DataFrame, truncate_to_icd_version: int) -> DataFrame:
    filtered_empty = filter_empty_labels(df)
    if truncate_to_icd_version is not None:
        filtered_empty['labels'] = (
            filtered_empty['labels'].apply(truncate_labels_9 if truncate_to_icd_version == 9 else truncate_labels_10)
        )
    return filtered_empty


def filter_empty_labels(df: DataFrame) -> DataFrame:
    return df[df.labels.str.len().astype(bool)]


def truncate_labels_9(labels: list[str]) -> list[str]:
    return [label[:4] if label.startswith('E') else label[:3] for label in labels]


def truncate_labels_10(labels: list[str]) -> list[str]:
    return [label[:3] for label in labels]


def compute_label_idx(*dfs: DataFrame) -> dict[str, int]:
    labels: pd.Series = pd.concat([df['labels'] for df in dfs], ignore_index=True)
    label_distribution = labels.explode().value_counts()
    label_idx = label_distribution.reset_index().labels.reset_index().set_index('labels')['index'].to_dict()
    return label_idx


class ClassificationCollator:
    def __init__(self, config: PretrainedConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path, config=config,
                                                       model_max_length=config.max_position_embeddings)

    def __call__(self, data):
        admission_notes = [x['admission_note'] for x in data]
        tokenized = self.tokenizer(admission_notes, padding=True, truncation=True)

        input_ids = [torch.tensor(x) for x in tokenized["input_ids"]]
        attention_masks = [torch.tensor(x, dtype=torch.bool) for x in tokenized["attention_mask"]]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks,
                                                          batch_first=True,
                                                          padding_value=0)

        labels = torch.stack([x['labels'] for x in data])
        lengths = torch.tensor([len(x) for x in input_ids])
        query_idces = torch.tensor([x['first_code'] for x in data]).unsqueeze(1).expand_as(labels).contiguous()

        return {"input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels,
                "lengths": lengths,
                'query_idces': query_idces,
                'first_codes': torch.tensor([x['first_code'] for x in data]),
                'raw_labels': [x['raw_labels'] for x in data]
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
        hadm_id = example['hadm_id']

        label_ids = [self.label2id[x] for x in labels]
        label_idxs = torch.tensor([x for x in label_ids], dtype=torch.int)
        labels = torch.zeros(len(self.label2id), dtype=torch.float32)
        labels[label_idxs] = 1

        return {'admission_note': note,
                'labels': labels,
                'hadm_id': hadm_id,
                'first_code': label_ids[0],
                'idx': idx,
                'raw_labels': example['labels']
                }
