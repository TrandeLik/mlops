import logging
import torch
from torch.optim import Optimizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import PreTrainedModel
from torch.utils.data import DataLoader


def get_optimizer(model: PreTrainedModel, config: dict) -> Optimizer:
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )
    logging.info(f"Optimizer created: AdamW with lr={lr}, weight_decay={weight_decay}")
    return optimizer
    

def train_one_epoch(model, dataloader, optimizer, device, logging_steps):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        if (i + 1) % logging_steps == 0:
            logging.info(f"Step {i+1}/{len(dataloader)}, Loss: {total_loss / (i + 1):.4f}")

    return total_loss / len(dataloader)


def predict(model: PreTrainedModel, dataloader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            labels = batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu())
            if labels is not None:
                all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits)
    if len(all_labels) > 0:
        all_labels = torch.cat(all_labels)
    else:
        all_labels = None

    return all_logits, all_labels


def evaluate(model: PreTrainedModel, dataloader: DataLoader, device: torch.device) -> tuple[float, str]:
    logits, labels = predict(model, dataloader, device)
    if labels is None:
        logging.warning("No labels found in dataloader. Cannot evaluate.")
        return 0.0, "No labels provided for evaluation."
    preds = torch.argmax(logits, dim=-1).numpy()
    labels = labels.numpy()
    accuracy = accuracy_score(labels, preds)
    report = classification_report(
        labels, 
        preds, 
        target_names=model.config.id2label.values(), 
        zero_division=0
    )
    return accuracy, report


def run_training_loop(model, train_loader, val_loader, optimizer, config: dict, device: torch.device) -> PreTrainedModel:
    num_epochs = config['training']['num_train_epochs']

    for epoch in range(num_epochs):
        logging.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, device, config['training']['logging_steps']
        )
        logging.info(f"Average training loss for epoch {epoch + 1}: {avg_train_loss:.4f}")

        accuracy, report = evaluate(model, val_loader, device)
        logging.info(f"Validation results for epoch {epoch + 1}:")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info("Classification Report:\n" + report)

    return model

