import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class ClassificationWithDescriptionCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, data):
        admission_notes = [x['admission_note'] for x in data]
        labels = torch.stack([x['labels'] for x in data])
        code_descriptions = [x['code_descriptions'] for x in data]
        num_descs_per_note = torch.tensor([len(x) for x in code_descriptions], dtype=torch.long)

        # flatten and tokenize code_descriptions
        # look at this cool double list comprehension no one understands and everyone has to google
        # Here is the link for you: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
        code_descriptions = [item for sublist in code_descriptions for item in sublist]
        tokenized_code_descriptions = self.tokenizer(code_descriptions,
                                                     padding=True,
                                                     truncation=True,
                                                     max_length=self.max_seq_len,
                                                     return_tensors="pt")

        tokenized = self.tokenizer(admission_notes,
                                   padding=False,
                                   truncation=True,
                                   max_length=self.max_seq_len,
                                   )
        input_ids = [torch.tensor(x) for x in tokenized["input_ids"]]
        attention_masks = [torch.tensor(x, dtype=torch.bool) for x in tokenized["attention_mask"]]
        lengths = torch.tensor([len(x) for x in input_ids])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks,
                                                          batch_first=True,
                                                          padding_value=0)

        embedding_ids = [x["label_idxs"] for x in data]
        max_embedding_id_count = torch.max(num_descs_per_note)

        embedding_id_attention_mask = torch.zeros((attention_masks.shape[0],
                                                   attention_masks.shape[1],
                                                   max_embedding_id_count), dtype=torch.bool)
        for i, num_descs in enumerate(num_descs_per_note):
            embedding_id_attention_mask[i, :, 0:num_descs] = 1

        mask = torch.ones(labels.shape, dtype=torch.bool)
        for i, id in enumerate(embedding_ids):
            x = id - 1
            x = x[x >= 0]
            mask[i, x] = 0

        embedding_ids = torch.nn.utils.rnn.pad_sequence(embedding_ids,
                                                        batch_first=True,
                                                        padding_value=0)

        return {"input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels,
                "icd_embedding_ids": embedding_ids,
                "icd_embedding_attention_mask": embedding_id_attention_mask,
                "lengths": lengths,
                "code_description_tokens": tokenized_code_descriptions.input_ids,
                "code_description_attention_masks": tokenized_code_descriptions.attention_mask,
                "codes_per_note": num_descs_per_note,
                "mask": mask}


class ClassificationCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int = 512, all_examples_with_null: bool = False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.all_examples_with_null = all_examples_with_null

    def __call__(self, data):
        admission_notes = [x['admission_note'] for x in data]
        labels = torch.stack([x['labels'] for x in data])
        tokenized = self.tokenizer(admission_notes,
                                   padding=False,
                                   truncation=True,
                                   max_length=self.max_seq_len,
                                   )
        batch_size = len(data)
        num_labels = len(data[0]['labels'])

        input_ids = [torch.tensor(x) for x in tokenized["input_ids"]]
        attention_masks = [torch.tensor(x, dtype=torch.bool) for x in tokenized["attention_mask"]]
        lengths = torch.tensor([len(x) for x in input_ids])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks,
                                                          batch_first=True,
                                                          padding_value=0)

        query_idces = torch.arange(batch_size).repeat_interleave(num_labels).view([batch_size, num_labels])

        return {"input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels,
                "lengths": lengths,
                'query_idces': query_idces,
                'first_codes': torch.tensor([x['first_code'] for x in data])
                }


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 examples,
                 label_lookup,
                 label_distribution,
                 num_codes_to_add: int = 0,
                 sampling_strategy: str = "random",
                 use_code_descriptions: bool = False,
                 add_null: bool = False):
        # tokenize admission notes
        self.examples = examples
        self.label_lookup = label_lookup
        self.inverse_label_lookup = {v: k for k, v in label_lookup.items()}
        self.num_codes_to_add = num_codes_to_add
        self.label_distribution = label_distribution
        self.sampling_strategy = sampling_strategy
        self.use_code_descriptions = use_code_descriptions
        self.add_null = add_null

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        note = example['text']
        labels = example['labels']
        hadm_id = example['hadm_id']

        label_ids = [self.label_lookup[x] for x in labels]
        label_idxs = torch.tensor([int(x) for x in label_ids])
        labels = torch.zeros(len(self.label_lookup), dtype=torch.float32)
        labels[label_idxs] = 1

        return {"admission_note": note,
                "labels": labels,
                # "label_idxs": annotated_labels,
                # "code_descriptions": description_texts,
                "hadm_id": hadm_id,
                'first_code': label_ids[0]}


def load_data_from(path: Path, glob: str):
    files = list(path.glob(glob))
    logging.warning(f'Detected {len(files)} files for glob {glob}: {files}')
    dfs = [pd.read_csv(file) if file.suffix == '.csv' else
           pd.read_parquet(file) if file.suffix == '.parquet' else None
           for file in files]
    for df in dfs:
        if 'test' in df:
            df.drop(columns='test', inplace=True)
        if 'SHORT_CODES' in df and 'labels' not in df:
            df.rename(columns={'SHORT_CODES': 'labels'}, inplace=True)
        if isinstance(df.labels.iloc[0], str):
            df.labels = df.labels.str.replace(r"[\[\]' ]", "", regex=True).str.split(",")
        df.text = df.text.str.strip()
    return pd.concat(dfs)


class MIMICClassificationDataModule(LightningDataModule):
    def __init__(self,
                 use_code_descriptions: bool = False,
                 data_dir: Path = Path("../../data"),
                 batch_size: int = 4,
                 eval_batch_size: int = 4,
                 tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 num_workers: int = 0,
                 num_codes_to_add: int = 0,
                 num_codes_to_add_val: int = 0,
                 sampling_strategy: str = "random",
                 val_sampling_strategy: str = "random",
                 max_seq_len: int = 512,
                 add_null_everywhere: bool = False):
        super().__init__()
        self.save_hyperparameters()

        training_data = load_data_from(data_dir, '*train*')
        test_data = load_data_from(data_dir, '*test*')
        validation_data = load_data_from(data_dir, '*val*')

        # build label index
        label_distribution = pd.concat([training_data.labels.explode(), validation_data.labels.explode(), test_data.labels.explode()]).value_counts()
        label_idx = label_distribution.reset_index().labels.reset_index().set_index('labels')['index'].to_dict()

        self.training_data = training_data.to_dict('records')
        self.test_data = test_data.to_dict('records')
        self.val_data = validation_data.to_dict('records')
        self.use_code_descriptions = use_code_descriptions
        self.label_idx = label_idx
        self.distribution = torch.from_numpy(label_distribution.values)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.add_null_everywhere = add_null_everywhere
        if self.use_code_descriptions:
            self.collator = ClassificationWithDescriptionCollator(self.tokenizer, max_seq_len)
        else:
            self.collator = ClassificationCollator(self.tokenizer, max_seq_len, all_examples_with_null=self.add_null_everywhere)

        self.num_workers = num_workers
        self.num_codes_to_add = num_codes_to_add
        self.num_codes_to_add_val = num_codes_to_add_val
        self.sampling_strategy = sampling_strategy
        self.val_sampling_strategy = val_sampling_strategy
        self.num_labels = len(label_idx)

    def setup(self, stage: Optional[str] = None):
        mimic_train = ClassificationDataset(self.training_data,
                                            label_lookup=self.label_idx,
                                            label_distribution=self.distribution,
                                            num_codes_to_add=self.num_codes_to_add,
                                            sampling_strategy=self.sampling_strategy,
                                            use_code_descriptions=self.use_code_descriptions,
                                            add_null=self.add_null_everywhere
                                            )

        mimic_val = ClassificationDataset(self.val_data,
                                          label_lookup=self.label_idx,
                                          label_distribution=self.distribution,
                                          num_codes_to_add=self.num_codes_to_add_val,
                                          sampling_strategy=self.val_sampling_strategy,
                                          use_code_descriptions=self.use_code_descriptions,
                                          add_null=self.add_null_everywhere)

        mimic_test = ClassificationDataset(self.test_data,
                                           label_lookup=self.label_idx,
                                           label_distribution=self.distribution,
                                           num_codes_to_add=self.num_codes_to_add_val,
                                           sampling_strategy=self.val_sampling_strategy,
                                           use_code_descriptions=self.use_code_descriptions,
                                           add_null=self.add_null_everywhere)
        self.mimic_train = mimic_train
        self.mimic_val = mimic_val
        self.mimic_test = mimic_test
        print("Val length: ", len(self.mimic_val))
        print("Train Length: ", len(self.mimic_train))

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
