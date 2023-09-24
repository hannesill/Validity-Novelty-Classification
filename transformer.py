import argparse
import datetime
from sklearn.metrics import f1_score, accuracy_score
from data.dataset import ClassificationDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import torch
from torch.utils.data import Dataset
import numpy as np


class TransformerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average='macro')
    }


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train the model")
    parser.add_argument("--filter", type=str, default="defeasible", help="Filter to use for the data")
    parser.add_argument("--augment", action="store_true", help="Whether to augment the data or not")
    args = parser.parse_args()

    # Check arguments
    assert args.filter in ["defeasible", "majority", "confident"], \
        "Invalid filter keyword. Valid options are: defeasible, majority, confident"

    # Hyperparameters
    MAX_LENGTH = 200
    EPOCHS = args.epochs

    # Set random seeds and device
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    # Preprocess data and train models for each task separately
    print("Training models...")
    for task in ["Validity", "Novelty"]:
        print("Task:", task)

        # Load datasets
        print("Loading data...")
        dataset_train = ClassificationDataset("data/TaskA_train.csv",
                                              task=task,
                                              augment=args.augment,
                                              filter_out=args.filter)
        dataset_valid = ClassificationDataset("data/TaskA_dev.csv", task=task)
        print("Train size:", len(dataset_train))
        print("Valid size:", len(dataset_valid))

        # Encode the sentences
        print("Encoding sentences...")
        encodings_train = tokenizer(
            dataset_train.sentences,
            add_special_tokens=True,
            truncation="longest_first",
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
            verbose=False
        )

        encodings_valid = tokenizer(
            dataset_valid.sentences,
            add_special_tokens=True,
            truncation="longest_first",
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
            verbose=False
        )

        # Create the model
        model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)

        # Convert the labels to tensors
        labels_train = torch.tensor(dataset_train.labels)
        labels_valid = torch.tensor(dataset_valid.labels)

        # Create the datasets
        transformer_dataset_train = TransformerDataset(encodings_train, labels_train)
        transformer_dataset_valid = TransformerDataset(encodings_valid, labels_valid)

        # Train the model
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=transformer_dataset_train,
            eval_dataset=transformer_dataset_valid
        )

        trainer.train()

        # Save the model
        trainer.model.eval()
        trainer.save_model(f"models/transformer-baseline_{task}_{timestamp}.pt")

        # Test the model on the test set
        print("Testing model")
        dataset_test = ClassificationDataset("data/TaskA_test.csv", task=task)

        # Encode the sentences
        encodings_test = tokenizer(
            dataset_test.sentences,
            add_special_tokens=True,
            truncation="longest_first",
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
            verbose=False
        )

        # Predict the labels
        transformer_dataset_test = TransformerDataset(encodings_test, torch.tensor(dataset_test.labels))

        # Calculate and print the accuracy and F1 score
        results = trainer.evaluate(transformer_dataset_test)
        print(f"Accuracy for {task}:", results["eval_accuracy"])
        print(f"F1 score for {task}:", results["eval_f1"])


