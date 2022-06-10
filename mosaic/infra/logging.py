# Importing stock libraries
import torch

# WandB – Import the wandb library
import wandb

import logging

logger = logging.getLogger(__name__)


def log_eval(tokenizer, model, device, loader):
    model.eval()
    total_loss = 0
    loss_count = 0
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            # TODO: fix the evaluation function - current implementation does not make sense
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)
            outputs = model(input_ids=ids, attention_mask=mask, labels=ids)

            loss = outputs.loss
            total_loss += loss.item()
            loss_count += 1

    eval_loss = total_loss / loss_count
    logger.info("Eval Loss: {}".format(eval_loss))
    wandb.log({"Eval Loss": eval_loss})
    return eval_loss
