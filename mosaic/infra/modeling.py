# Importing stock libraries
import torch

# Import os for env varibles via Beaker
import re

# WandB – Import the wandb library
import wandb
import logging
from tqdm import tqdm

logger = logging.getLogger("modeling")
from mosaic.infra.logging import log_eval


def train(
    epoch,
    tokenizer,
    model,
    device,
    loader,
    optimizer,
    val_loader=None,
    metric_json=None,
):

    model.train()
    batch_count = len(loader)

    for iteration, data in tqdm(enumerate(loader, 0)):
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)
        outputs = model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = outputs.loss
        step = iteration + epoch * len(loader)
        if iteration % 100 == 0:
            batches_left = batch_count - iteration
            logger.info(
                f"\nEpoch: {epoch}, Iteration: {iteration}, Loss:  {loss.item()}, Batches left: {batches_left}"
            )
            wandb.log(
                {
                    "Training Loss": loss.item(),
                    "Epoch": epoch,
                    "Batches left": batch_count - iteration,
                }
            )
            if metric_json:
                metric_json["train"].append({"train_loss": loss.item(), "step": step})

        if iteration % 500 == 0:
            logger.info(
                f"\nEpoch: {epoch}, Loss:  {loss.item()}, BatchesLeft: {batches_left}"
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0 and val_loader != None:
            val_loss = log_eval(tokenizer, model, device, val_loader)
            if metric_json:
                metric_json["valid"].append({"valid_loss": val_loss, "step": step})
            model.train()


def validate(tokenizer, model, device, loader, max_length=50):
    model.eval()
    predictions = []
    actuals = []
    sources = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=True,
                max_length=max_length,
                num_beams=5,
                top_k=50,
                top_p=0.95,
            )

            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for t in y
            ]
            source = [
                tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for s in ids
            ]

            if _ % 100 == 0:
                logger.info(f"Completed {_}")

            sources.extend(source)
            predictions.extend(preds)
            actuals.extend(target)
    return sources, predictions, actuals


def beam_generations(tokenizer, model, device, loader, top_k=40, max_length=50):
    # This method assumes batch size of 1
    model.eval()
    records = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                temperature=1.0,
                do_sample=False,
                max_length=max_length,
                top_p=0.9,
                top_k=top_k,
                repetition_penalty=1.0,
                num_return_sequences=10 if top_k > 1 else 1,
                num_beams=10,
            )

            source = [
                tokenizer.decode(s, clean_up_tokenization_spaces=True) for s in ids
            ]
            preds = [
                tokenizer.decode(g, clean_up_tokenization_spaces=True)
                for g in generated_ids
            ]

            head_event = re.findall(r"(.*)\[GEN\]", source[0])[0].strip()
            head = " ".join(head_event.split(" ")[:-1]).strip()
            relation = head_event.split(" ")[-1].strip()
            preds = [
                re.findall(r"\[GEN\](.*?)\[EOS\]", pred)[0].strip()
                for pred in preds
                if len(re.findall(r"\[GEN\](.*?)\[EOS\]", pred))
            ]
            records.append(
                {
                    "head": head,
                    "relation": relation,
                    "tails": [],
                    "generations": preds,
                }
            )
            if _ % 100 == 0:
                logger.info(f"Completed {_}")

    return records
