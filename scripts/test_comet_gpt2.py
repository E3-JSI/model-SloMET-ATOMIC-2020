import os
import re
import yaml
import json
import argparse

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

# Importing the GPT2 modules from huggingface/transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import write_items
from mosaic.infra.modeling import beam_generations
from mosaic.datasets.KGDataset import KGDataset

# for typing the objects
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================================
# Logging configuration
# ===============================================

import logging

logger = logging.getLogger("gpt2_comet_atomic")
logging.basicConfig(level=logging.INFO)

# logger.info for allenai beaker verification
logger.info(device)
logger.info(torch.cuda.device_count())

# ===============================================
# Helper functions
# ===============================================


def create_mapping_key(head, relation):
    return f"{re.sub(' ', '', head)} @@ {relation}"


def retrieve_ref_data(test_file_path):
    df = pd.read_csv(test_file_path, encoding="utf8", sep="\t")
    return {
        create_mapping_key(row.Index[0], row.Index[1]): list(
            filter(lambda x: not pd.isnull(x), row.tail_event)
        )
        for row in df.groupby(["head_event", "relation"]).agg(list).itertuples()
    }


class Config:
    """Configuration parameters container class"""

    pass


# ===============================================
# Import parameters
# ===============================================

params = yaml.safe_load(open("params.yaml"))

# ===============================================
# Define the main script
# ===============================================


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_path", type=str, help="The path to the test data"
    )
    parser.add_argument(
        "--models_dir_path",
        type=str,
        help="The path to the directory where the model is stored",
    )
    parser.add_argument(
        "--results_dir_path",
        type=str,
        help="The path to the directory where the results are stored",
    )
    args = parser.parse_args()

    test_data_path = args.test_data_path
    models_dir_path = args.models_dir_path
    results_dir_path = args.results_dir_path

    # ===========================================
    # Configure the process
    # ===========================================

    # replace wandb with a generic config class (we do not use it here)
    config = Config()
    config.SEED = int(params["train"]["SEED"])
    config.TEST_TOP_K = int(params["test"]["TEST_TOP_K"])
    config.TEST_BATCH_SIZE = 1

    config.IN_LEN = int(params["model"]["IN_LEN"])
    config.OUT_LEN = int(params["model"]["OUT_LEN"])

    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # ===========================================
    # Prepare the model and tokenizer
    # ===========================================

    logging.info("Loading model from {}".format(models_dir_path))

    # initialize the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(models_dir_path)
    tokenizer = AutoTokenizer.from_pretrained(models_dir_path)

    logging.info("Move model to device {}".format(device))
    model = model.to(device)

    # ===========================================
    # Prepare the test set
    # ===========================================

    # prepare the test dataset
    test_dataset = pd.read_csv(test_data_path, encoding="utf8", sep="\t")
    test_dataset = test_dataset.drop_duplicates(
        ["head_event", "relation"], ignore_index=True
    )
    test_dataset.head_event = (
        test_dataset.head_event + " " + test_dataset.relation + " [GEN]"
    )
    test_dataset.tail_event = test_dataset.tail_event + " [EOS]"
    test_set = KGDataset(
        dataframe=test_dataset,
        tokenizer=tokenizer,
        source_len=config.IN_LEN,
        summ_len=config.OUT_LEN,
        is_eval=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # ===========================================
    # Start the training process
    # ===========================================

    # generate the predictions and write them into a file
    pred_generations = beam_generations(
        tokenizer=tokenizer,
        model=model,
        device=device,
        loader=test_loader,
        top_k=config.TEST_TOP_K,
        max_length=config.OUT_LEN,
    )

    # get the reference mapping and enrich the predictions with it
    tails_mapping = retrieve_ref_data(test_data_path)

    def prepare_json_object(r):
        return json.dumps(
            {
                **r,
                "tails": tails_mapping[create_mapping_key(r["head"], r["relation"])],
            },
            ensure_ascii=False,
        )

    write_items(
        os.path.join(results_dir_path, "pred_generations.jsonl"),
        [prepare_json_object(r) for r in pred_generations],
    )


if __name__ == "__main__":
    main()
