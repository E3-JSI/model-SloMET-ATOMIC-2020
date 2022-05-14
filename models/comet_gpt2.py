import os
import yaml
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from optparse import OptionParser

# Importing the GPT2 modules from huggingface/transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# WandB – Import the wandb library
import wandb

from split.utils import write_items
from mosaic.infra.modeling import train, validate, beam_generations
from mosaic.datasets.KGDataset import KGDataset

# for typing the objects
from typing import List

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===============================================
# Logging configuration
# ===============================================

import logging
logger = logging.getLogger("gpt2-comet")
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

def main():
    wandb.init(project="gpt2_comet_atomic")

    config = wandb.config
    config.TRAIN_BATCH_SIZE = int(params["train"]["TRAIN_BATCH_SIZE"])
    config.VALID_BATCH_SIZE = int(params["train"]["VALID_BATCH_SIZE"])
    config.TRAIN_EPOCHS = int(params["train"]["TRAIN_EPOCHS"])
    config.VALID_EPOCHS = int(params["train"]["VALID_EPOCHS"])
    config.LEARNING_RATE = float(params["train"]["LEARNING_RATE"])
    config.SEED = int(params["train"]["SEED"])
    config.SUMMARY_LEN = 0 # Used for t5
    config.OUT_DIR = params["train"]["OUT_DIR"]
    config.DO_TRAIN = params["train"]["DO_TRAIN"]
    config.DO_PRED = params["train"]["DO_TRAIN"]
    config.PRED_FILE = params["train"]["PRED_FILE"]
    config.TEST_TOP_K = int(params["test"]["TEST_TOP_K"])
    config.TEST_BATCH_SIZE = int(params["test"]["TEST_BATCH_SIZE"])


    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # initialize the model and tokenizer
    model_name = params["model"]["model_name"]
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'eos_token': '[EOS]',
        'additional_special_tokens': [
            'LocationOfAction',
            'HinderedBy',
            'HasFirstSubevent',
            'NotHasProperty',
            'NotHasA',
            'HasA',
            'AtLocation',
            'NotCapableOf',
            'CausesDesire',
            'HasPainCharacter',
            'NotDesires',
            'MadeUpOf',
            'InstanceOf',
            'SymbolOf',
            'xReason',
            'isAfter',
            'HasPrerequisite',
            'UsedFor',
            'MadeOf',
            'MotivatedByGoal',
            'Causes',
            'oEffect',
            'CreatedBy',
            'ReceivesAction',
            'NotMadeOf',
            'xWant',
            'PartOf',
            'DesireOf',
            'HasPainIntensity',
            'xAttr',
            'DefinedAs',
            'oReact',
            'xIntent',
            'HasSubevent',
            'oWant',
            'HasProperty',
            'IsA',
            'HasSubEvent',
            'LocatedNear',
            'Desires',
            'isFilledBy',
            'isBefore',
            'InheritsFrom',
            'xNeed',
            'xEffect',
            'xReact',
            'HasLastSubevent',
            'RelatedTo',
            'CapableOf',
            'NotIsA',
            'ObjectUse',
            '[GEN]'
        ]
    })

    # ===========================================
    # Prepare the data sets
    # ===========================================

    # prepare the training dataset
    train_dataset = pd.read_csv(params["train"]["TRAIN_DATA_PATH"], encoding='utf-8', sep="\t")
    train_dataset.head_event = train_dataset.head_event + ' ' + train_dataset.relation + " [GEN]"
    train_dataset.tail_event = train_dataset.tail_event + ' [EOS]'

    logger.info(train_dataset.head())
    logger.info(train_dataset.tail_event)

    # prepare the validation dataset
    val_dataset = pd.read_csv(params["train"]["VALID_DATA_PATH"], encoding='utf-8', sep="\t")
    val_dataset = val_dataset[['head_event', 'tail_event', 'relation']]
    val_dataset.head_event = val_dataset.head_event + ' ' + val_dataset.relation + " [GEN]"
    val_dataset.tail_event = val_dataset.tail_event + ' [EOS]'

    logger.info(val_dataset.tail_event)
    logger.info(val_dataset.head())

    # prepare the test dataset
    test_dataset = pd.read_csv(params["test"]["TEST_DATA_PATH"], encoding='utf-8', sep="\t")
    test_dataset = test_dataset[['head_event', 'tail_event', 'relation']]
    test_dataset.head_event = test_dataset.head_event + ' ' + test_dataset.relation + " [GEN]"
    test_dataset.tail_event = test_dataset.tail_event + ' [EOS]'

    logger.info(test_dataset.tail_event)
    logger.info(test_dataset.head())

    # ############################SETTING IN_LEN


    #FIGURE OUT MAX LENGTH FROM TOKENIZER FROM FILE KGDATASET.PY IN MOSAIC LINE 46 ON GITHUB
    #GO THRU IT RUN TOKENIZER WITHOUT MAX LENGTH PARAMS TO TRUNCATE AND
    #######
    print("//////////////////////////////////////////////////")
    print("GET IN LEN")
    from transformers import AutoTokenizer, AutoModelWithLMHead

    TestInLenTokenizer= AutoTokenizer.from_pretrained('macedonizer/sl-gpt2')

    MxInLen=0
    TrainHeadEventL = train_dataset['head_event'].tolist()
    for Str in TrainHeadEventL:
      if not isinstance(Str, str):
        Str = str(Str)
      encoded_input = TestInLenTokenizer(Str, return_tensors="pt")
      CurInLen = encoded_input["input_ids"].size(dim=1)
      if CurInLen > MxInLen:
        MxInLen = CurInLen


    # #input_text = 'Ljubljana Maribor Hello Test'#SET THIS TO YOUR INPUT THTA YOU WANT TO PUT IN MODEL FROM THE DATASET
    # ##THIS ENCODED INPUT IS A PYTORCH TENSOR
    # #PRINTS LENGTH OF ENCODED INPUT, TAKE MAX OF ALL THESE (DON'T NEED TO GO THRU WHOLE DATASET, GO THRU A SAMPLE)
    # #encoded_input = tokenizer(input_text, return_tensors="pt")
    # #print(type(encoded_input))
    # #CurInLen = encoded_input["input_ids"].size(dim=1)
    # #print(type(encoded_input["input_ids"].shape))
    print("IN_LEN: ", MxInLen)
    # print("//////////////////////////////////////////////////")
    # #########################################################

    #ORIGINAL
    # config.IN_LEN = int(os.environ.get("IN_LEN", 16))

    #ADRIAN
    config.IN_LEN = MxInLen +2

    # ORIGINAL
    config.OUT_LEN = int(os.environ.get("OUT_LEN", config.IN_LEN + 44))
    # ADRIAN
    # OUTPUT OF MODEL IS THE INPUT + OUTPUT TOGETHER (check KGDataset inputs)
    # 16 was original IN_LEN, 34 was original OUT_LEN, so 34-16 was actual length of model generations
    # config.OUT_LEN = config.IN_LEN + (34-16)

    #############################



    logger.info("TRAIN Dataset tuple count: {}".format(train_dataset.shape))
    logger.info("DEV Dataset tuple_count: {}".format(val_dataset.shape))

    training_set = KGDataset(train_dataset, tokenizer, config.OUT_LEN, config.SUMMARY_LEN, model="gpt2")
    val_set = KGDataset(val_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
    val_set_mini = KGDataset(val_dataset.head(2000), tokenizer, config.IN_LEN,  config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
    test_set = KGDataset(test_dataset, tokenizer, config.IN_LEN,  config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)

    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params, drop_last=True)
    val_loader_mini = DataLoader(val_set_mini, **val_params, drop_last=True)

    # ===========================================
    # Prepare the model
    # ===========================================

    logging.info("Loading model from {}".format(model_name))

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    logging.info("Move model to device {}".format(device))
    model = model.to(device)

    wandb.watch(model, log="all")

    # ===========================================
    # Initialize the optimizer
    # ===========================================

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    if config.DO_TRAIN:

        print("////////////////////////////////////////////////////")
        print("TRAINING BEGUN")
        print("////////////////////////////////////////////////////")

        logger.info('Initiating Fine-Tuning for the model on our dataset')

        for epoch in range(config.TRAIN_EPOCHS):
            train(epoch, tokenizer, model, device, training_loader, optimizer, val_loader_mini, model_class="gpt2")
            model.save_pretrained('{}/checkpoint_{}'.format(config.OUT_DIR, epoch))
            tokenizer.save_pretrained('{}/checkpoint_{}'.format(config.OUT_DIR, epoch))
        model.save_pretrained('{}/models'.format(config.OUT_DIR))

    if config.DO_PRED:

        print("////////////////////////////////////////////////////")
        print("PREDICTION GENERATION BEGUN")
        print("////////////////////////////////////////////////////")


        if config.PRED_FILE.endswith("jsonl"):
            records = read_jsonl_lines(config.PRED_FILE)
            pred_dataset = pd.DataFrame.from_records(records)
            pred_dataset = pred_dataset.rename(columns={"head": "head_event", "tails": "tail_event"})
            pred_dataset = pred_dataset.explode('tail_event')
        else:
            pred_dataset = pd.read_csv(config.PRED_FILE, encoding="utf-8", sep="\t")


        pred_dataset = pred_dataset.drop_duplicates(['head_event', 'relation'], ignore_index=True)

        pred_dataset.head_event = pred_dataset.head_event + ' ' + pred_dataset.relation + " [GEN]"
        pred_dataset.tail_event = pred_dataset.tail_event + ' [EOS]'
        logger.info(pred_dataset.tail_event)
        logger.info(pred_dataset.head())

        pred_set = KGDataset(pred_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
        pred_loader = DataLoader(pred_set, **val_params, drop_last=False)

        pred_generations = beam_generations(tokenizer, model, device, pred_loader, top_k=config.TEST_TOP_K)
        write_items(os.path.join(config.OUT_DIR, "pred_generations.jsonl"),
                    [json.dumps(r) for r in pred_generations])

        # Resave the model to keep generations and model associated
        model.save_pretrained('{}/models'.format(config.OUT_DIR))
        tokenizer.save_pretrained('{}/models'.format(config.OUT_DIR))

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-t", "--test_install",
                      action="store_true", default=False,
                      help="Test install, without running any modeling code.")

    (options, args) = parser.parse_args()
    if not options.test_install:
        main()
