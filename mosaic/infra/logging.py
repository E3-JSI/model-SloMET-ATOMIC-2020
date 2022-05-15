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
            y = data['target_ids'].to(device, dtype = torch.long)
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            outputs = model(input_ids = ids, attention_mask = mask, labels=ids)

            loss = outputs[0]
            total_loss += loss.item()
            loss_count += 1

    logger.info("Eval Loss: {}".format(total_loss / loss_count))
    wandb.log({"Eval Loss": total_loss / loss_count})