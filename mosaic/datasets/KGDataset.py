# Importing stock libraries
import torch
from torch.utils.data import Dataset

import logging

logger = logging.getLogger("gpt2-comet")
logging.basicConfig(level=logging.DEBUG)


class KGDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len=0, is_eval=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.head_event
        self.ctext = self.data.tail_event
        self.is_eval = is_eval

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        ctext = str(self.ctext[index])
        ctext = " ".join(ctext.split())

        if self.is_eval:
            source = self.tokenizer(
                [text],
                add_special_tokens=False,
                max_length=self.source_len,
                truncation="longest_first",
                padding="do_not_pad",
                return_tensors="pt",
            )
            target = self.tokenizer(
                [ctext],
                add_special_tokens=False,
                max_length=self.summ_len,
                truncation="longest_first",
                padding="do_not_pad",
                return_tensors="pt",
            )
        else:
            source = self.tokenizer(
                [text + " " + ctext],
                add_special_tokens=False,
                max_length=self.source_len + self.summ_len,
                truncation="longest_first",
                padding="max_length",
                return_tensors="pt",
            )
            target = source

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_mask.to(dtype=torch.long),
        }
