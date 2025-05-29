from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import evaluate
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import argparse


def setup_distributed():
    # Check if distributed environment variables are set
    if all(key in os.environ for key in ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']):
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # Use global rank for GPU assignment to ensure unique GPU per rank
        gpu = rank % torch.cuda.device_count()

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Using GPU {gpu} for rank {rank}")

        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl',
                              init_method='env://',
                              world_size=world_size,
                              rank=rank)
        return rank, world_size, gpu, True
    else:
        # Single GPU training
        return 0, 1, 0, False

def collate_fn(batch):
    return {
        'input_ids': torch.tensor([item['input_ids'] for item in batch]),
        'attention_mask': torch.tensor([item['attention_mask'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch])
    }

def train_epoch(model, train_loader, optimizer, criterion, device, is_distributed):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if is_distributed:
        dist.all_reduce(torch.tensor(total_loss).to(device))
        total_loss /= dist.get_world_size()

    return total_loss / len(train_loader)

def evaluate_model(model, eval_loader, criterion, device, is_distributed):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if is_distributed:
        dist.all_reduce(torch.tensor(total_loss).to(device))
        total_loss /= dist.get_world_size()

    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    return total_loss / len(eval_loader), accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()

    # Initialize distributed training
    rank, world_size, gpu, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    print("rank", rank, "world_size", world_size, "gpu", gpu, "is_distributed", is_distributed)

    # Load and process dataset
    dataset = load_dataset("yelp_review_full")
    dataset["train"] = dataset["train"].select(range(10000))
    dataset["test"] = dataset["test"].select(range(128))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Create data loaders
    if is_distributed:
        train_sampler = DistributedSampler(tokenized_datasets["train"])
        eval_sampler = DistributedSampler(tokenized_datasets["test"])
        train_loader = DataLoader(
            tokenized_datasets["train"],
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn
        )
        eval_loader = DataLoader(
            tokenized_datasets["test"],
            batch_size=args.batch_size,
            sampler=eval_sampler,
            collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            tokenized_datasets["train"].shuffle(seed=42),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        eval_loader = DataLoader(
            tokenized_datasets["test"].shuffle(seed=42),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[gpu])

    # Setup optimizer and loss
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()

    # Training loop
    for epoch in range(args.num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, is_distributed)
        eval_loss, accuracy = evaluate_model(model, eval_loader, criterion, device, is_distributed)

        if rank == 0:  # Only print on main process
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Evaluation Loss: {eval_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")

    # Cleanup distributed training
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
