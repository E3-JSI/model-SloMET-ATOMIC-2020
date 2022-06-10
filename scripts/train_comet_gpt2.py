import yaml
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# Importing the GPT2 modules from huggingface/transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# WandB – Import the wandb library
import wandb

from mosaic.infra.modeling import train
from mosaic.datasets.KGDataset import KGDataset

# for typing the objects
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================================
# Logging configuration
# ===============================================

import logging

logger = logging.getLogger("gpt2_comet_atomic")
logging.basicConfig(level=logging.DEBUG)

# logger.info for allenai beaker verification
logger.info(device)
logger.info(torch.cuda.device_count())

# ===============================================
# Helper functions
# ===============================================


def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


# ===============================================
# Import parameters
# ===============================================

params = yaml.safe_load(open("params.yaml"))

# ===============================================
# Define the main script
# ===============================================

USE_WANDB = False


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path", type=str, help="The path to the training data"
    )
    parser.add_argument(
        "--valid_data_path", type=str, help="The path to the validation data"
    )
    parser.add_argument(
        "--models_dir_path",
        type=str,
        help="The path to the directory where the model is stored",
    )
    args = parser.parse_args()

    # get the script arguments
    train_data_path = args.train_data_path
    valid_data_path = args.valid_data_path
    models_dir_path = args.models_dir_path

    # ===========================================
    # Configure the process
    # ===========================================

    wandb.init(project="gpt2_comet_atomic")

    config = wandb.config
    config.TRAIN_BATCH_SIZE = int(params["train"]["TRAIN_BATCH_SIZE"])
    config.VALID_BATCH_SIZE = int(params["train"]["VALID_BATCH_SIZE"])
    config.TRAIN_EPOCHS = int(params["train"]["TRAIN_EPOCHS"])
    config.VALID_EPOCHS = int(params["train"]["VALID_EPOCHS"])
    config.LEARNING_RATE = float(params["train"]["LEARNING_RATE"])
    config.SEED = int(params["train"]["SEED"])

    config.IN_LEN = int(params["model"]["IN_LEN"])
    config.OUT_LEN = int(params["model"]["OUT_LEN"])

    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # ===========================================
    # Prepare the tokenizer
    # ===========================================

    model_name = params["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "eos_token": "[EOS]",
            "additional_special_tokens": [
                "LocationOfAction",
                "HinderedBy",
                "HasFirstSubevent",
                "NotHasProperty",
                "NotHasA",
                "HasA",
                "AtLocation",
                "NotCapableOf",
                "CausesDesire",
                "HasPainCharacter",
                "NotDesires",
                "MadeUpOf",
                "InstanceOf",
                "SymbolOf",
                "xReason",
                "isAfter",
                "HasPrerequisite",
                "UsedFor",
                "MadeOf",
                "MotivatedByGoal",
                "Causes",
                "oEffect",
                "CreatedBy",
                "ReceivesAction",
                "NotMadeOf",
                "xWant",
                "PartOf",
                "DesireOf",
                "HasPainIntensity",
                "xAttr",
                "DefinedAs",
                "oReact",
                "xIntent",
                "HasSubevent",
                "oWant",
                "HasProperty",
                "IsA",
                "HasSubEvent",
                "LocatedNear",
                "Desires",
                "isFilledBy",
                "isBefore",
                "InheritsFrom",
                "xNeed",
                "xEffect",
                "xReact",
                "HasLastSubevent",
                "RelatedTo",
                "CapableOf",
                "NotIsA",
                "ObjectUse",
                "[GEN]",
            ],
        }
    )

    # ===========================================
    # Prepare the model
    # ===========================================

    logging.info("Loading model from {}".format(model_name))

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    logging.info("Move model to device {}".format(device))
    model = model.to(device)

    wandb.watch(model, log="all")

    # ===========================================
    # Prepare the training and validation sets
    # ===========================================

    # prepare the training dataset
    train_dataset = pd.read_csv(train_data_path, encoding="utf-8", sep="\t")
    train_dataset.head_event = (
        train_dataset.head_event + " " + train_dataset.relation + " [GEN]"
    )
    train_dataset.tail_event = train_dataset.tail_event + " [EOS]"
    training_set = KGDataset(
        dataframe=train_dataset, tokenizer=tokenizer, source_len=config.IN_LEN
    )
    training_loader = DataLoader(
        training_set,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    # prepare the validation dataset
    val_dataset = pd.read_csv(valid_data_path, encoding="utf-8", sep="\t")
    val_dataset = val_dataset[["head_event", "tail_event", "relation"]]
    val_dataset.head_event = (
        val_dataset.head_event + " " + val_dataset.relation + " [GEN]"
    )
    val_dataset.tail_event = val_dataset.tail_event + " [EOS]"
    val_set_mini = KGDataset(
        dataframe=val_dataset.head(2000),
        tokenizer=tokenizer,
        source_len=config.IN_LEN,
        summ_len=config.OUT_LEN,
        is_eval=True,
    )
    val_loader_mini = DataLoader(
        val_set_mini,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ===========================================
    # Prepare the optimizer
    # ===========================================

    # use weighted adam optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.LEARNING_RATE)

    logger.info("Initiating Fine-Tuning for the model on our dataset")

    # ===========================================
    # Prepare the loss evaluation
    # ===========================================

    train_metrics = {"train": [], "valid": []}

    # ===========================================
    # Start the training process
    # ===========================================

    for epoch in range(config.TRAIN_EPOCHS):
        train(
            epoch=epoch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            loader=training_loader,
            optimizer=optimizer,
            val_loader=val_loader_mini,
            metric_json=train_metrics,
        )
        # save the current model checkpoint
        model.save_pretrained("{}/checkpoint_{}".format(models_dir_path, epoch))
        tokenizer.save_pretrained("{}/checkpoint_{}".format(models_dir_path, epoch))
        # save the model as the latest checkpoint
        model.save_pretrained("{}/checkpoint_latest".format(models_dir_path))
        tokenizer.save_pretrained("{}/checkpoint_latest".format(models_dir_path))

    # store the metrics
    with open("plots/train_metrics.json", "w", encoding="utf8") as f:
        json.dump(train_metrics, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
