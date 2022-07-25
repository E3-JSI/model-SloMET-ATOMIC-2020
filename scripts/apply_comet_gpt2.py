import yaml
import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader

# Importing the GPT2 modules from huggingface/transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from mosaic.infra.modeling import beam_generations

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
# Import parameters
# ===============================================

params = yaml.safe_load(open("params.yaml"))


class Config:
    """Configuration parameters container class"""

    pass


# ===============================================
# Define the main script
# ===============================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models_dir_path",
        type=str,
        help="The path to the directory where the model is stored",
    )

    parser.add_argument("--head_event", type=str, help="The head event")

    parser.add_argument(
        "--relations",
        type=str,
        help="The path to the directory where the results are stored",
    )
    args = parser.parse_args()

    models_dir_path = args.models_dir_path
    head_event = args.head_event
    relations = args.relations

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
    # Convert the
    # ===========================================

    dataset = []
    for relation in relations.split(","):
        source = tokenizer(
            [f"{head_event} {relation} [GEN]"],
            add_special_tokens=False,
            max_length=50,  # change the maximum length
            truncation="longest_first",
            padding="do_not_pad",
            return_tensors="pt",
        )
        dataset.append(
            {
                "source_ids": source["input_ids"].squeeze().to(dtype=torch.long),
                "source_mask": source["attention_mask"].squeeze().to(dtype=torch.long),
            }
        )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
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
        loader=data_loader,
        top_k=config.TEST_TOP_K,
        max_length=config.OUT_LEN,
    )

    print(pred_generations)


if __name__ == "__main__":
    main()
