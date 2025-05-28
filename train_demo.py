
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, HfArgumentParser
import numpy as np
import evaluate

# Code from https://huggingface.co/docs/transformers/en/training

if __name__ == "__main__":

    # Load and process dataset
    dataset = load_dataset("yelp_review_full")
    dataset["train"] = dataset["train"].select(range(1000))
    dataset["test"] = dataset["test"].select(range(128))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

    # Load and process model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    # Load and process metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments and trainer
    # training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
    hf_parser = HfArgumentParser(TrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
