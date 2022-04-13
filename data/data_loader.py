# Import Glue dataset

from cProfile import label
from datetime import datetime
import pdb
from typing import Optional

import os
import datasets
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    seed_everything,
    Trainer,
)
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    glue_convert_examples_to_features,
)


AVAIL_GPUS = min(1, torch.cuda.device_count())

# Torch
class GLUE(Dataset):
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    task_file_map = {"sst2": "SST-2"}

    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    def __init__(
        self,
        data_load_path: str = "/cognitive_comp/yangqi/data/glue_data",
        # model_name_or_path: str ='./model/distilbert-base-uncased',
        model_name_or_path: str = "bert-base-cased",
        task_name: str = "sst2",
        max_seq_length: int = 64,
        set_name: str = "train",
        **kwargs
    ):
        super().__init__()
        self.data_load_path = data_load_path
        self.set_name = set_name
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name

        self.text_fields = self.task_text_field_map[task_name]
        self.max_seq_length = max_seq_length
        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.task_load_path = os.path.join(
            self.data_load_path, self.task_file_map[self.task_name]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )

        self.load_data()

    def __len__(self):

        return len(self.data)

    def load_data(self):
        print("load data")
        self.data = pd.read_csv(
            os.path.join(self.task_load_path, self.set_name + ".csv")
        )

    def __getitem__(self, idx):
        features = self.convert_to_features(idx)
        return features

    def convert_to_features(self, idx):
        sentence = self.data[self.text_fields[0]][idx]
        if "label" in self.data.keys():
            label = self.data["label"][idx]
        else:
            pdb.set_trace()
        features = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        # features[self.text_fields[0]] = sentence
        features["datasets_idx"] = idx

        features["labels"] = [label]
        return features


def collate_fn(batch):
    #
    # batch= list(zip(*batch))

    # print(batch)
    # labels = torch.tensor(batch[1],dtype=torch.int32)
    # texts = batch[0]
    # del batch
    x1, x2, x3, y = [], [], [], []
    for unit in batch:
        x1.append(unit["input_ids"])
        x2.append(unit["attention_mask"])
        x3.append(unit["token_type_ids"])
        y.extend(unit["labels"])
    # print (x1,x2,y)
    return torch.tensor(x1), torch.tensor(x2), torch.tensor(x3), torch.tensor(y)


class GLUEDataLoader(LightningDataModule):

    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    def __init__(self, task="sst2", train_batch_size=32, eval_batch_size=32, max_seq_length=128, **kwargs):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length
        self.task_name = task

        self.text_fields = self.task_text_field_map[task]
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        self.prepare_data()

    def prepare_data(self):
        task = self.task_name
        self.dataset = {
            "train": GLUE(task_name=task, set_name="train",max_seq_length=self.max_seq_length),
            "test": GLUE(task_name=task, set_name="test",max_seq_length=self.max_seq_length),
            "dev": GLUE(task_name=task, set_name="dev",max_seq_length=self.max_seq_length),
        }
        # AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        print("Prepare data")
        print("Tokenizer data")
        # pdb.set_trace()

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Return validation data"""
        return DataLoader(
            self.dataset["dev"], batch_size=self.eval_batch_size, collate_fn=collate_fn
        )

    def test_dataloader(self):
        """Return test data"""
        return DataLoader(
            self.dataset["dev"], batch_size=self.eval_batch_size, collate_fn=collate_fn
        )

    def discard_setup(self):
        """Set up tasks and split train and test"""
        # self.dataset = datasets.load_dataset('glue', self.task_name)

        for split in self.dataset.keys():
            for idx in range(len(self.dataset[split])):
                features = map(self.convert_to_features, self.data[split], batched=True)
                self.dataset[split][idx].update(features)
                # 我们用的 GLUE dataset 写死了 column
            # self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            # self.dataset[split].set_format(type="torch", columns=self.columns)

    def discard_convert_to_features2(self, examples):
        sentence = examples["sentence"]
        label = examples["train"]
        features = self.tokenizer(
            sentence, max_length=self.max_seq_length, padding=True, truncation=True
        )
        features["labels"] = label
        return features

    def discard_convert_to_features(self, example_batch, indices=None):
        """Return dataset consist of word,sentense为-level"""
        # 编码单个句子或句子对
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[self.text_fields[0]],
                    example_batch[self.text_fields[1]],
                )
            )
        else:
            texts_or_text_pairs = list(example_batch[self.text_fields[0]])

        # 文本/文本对词元化
        # 关于 batch encode plus https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus
        # https://huggingface.co/docs/transformers/main/en/pad_truncation
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
        )

        # 将label重命名为labels，以使其更容易传递给模型的forward方法
        features["label"] = example_batch["label"]
        return features


# Ligntning Module from huggingface
class GLUEDataModule(LightningDataModule):
    task_file_map = {"sst2": "SST-2"}

    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        data_load_path: str = "./data",
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.data_load_path = data_load_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.task_load_path = os.path.join(
            self.data_load_path, self.task_file_map[self.task_name]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )

    def setup(self, stage: str):
        """Set up tasks and split train and test"""
        # self.dataset = datasets.load_dataset('glue', self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_columns
            ]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def from_tsv_csv(self):
        for name in ["train", "test", "dev"]:
            df = pd.read_csv(self.task_load_path + "/" + name + ".tsv", sep="\t")
            df.to_csv(
                self.task_load_path + "/" + name + ".tsv",
                encoding="utf_8_sig",
                index=False,
            )
            print("Turn dataset tsv to csv...")

    def prepare_data(self):
        """Download tasks dataset and tokenizer"""
        # datasets.load_dataset('glue', self.task_name)
        # The code will return Connextion Error
        self.dataset = datasets.load_dataset(
            "csv",
            data_files={
                "train": self.task_load_path + "/train.csv",
                "test": self.task_load_path + "/dev.csv",
                "validation": self.task_load_path + "/dev.csv",
            },
        )
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    def val_dataloader(self):
        """Return validation data"""
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"], batch_size=self.eval_batch_size
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        """Return test data"""
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def convert_to_features(self, example_batch, indices=None):
        """Return dataset consist of word,sentense为-level"""
        # 编码单个句子或句子对
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[self.text_fields[0]],
                    example_batch[self.text_fields[1]],
                )
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # 文本/文本对词元化
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
        )

        # 将label重命名为labels，以使其更容易传递给模型的forward方法
        features["labels"] = example_batch["label"]

        return features
