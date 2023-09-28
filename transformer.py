import argparse
import datetime
import random
from sklearn.metrics import f1_score
from data.dataset import ClassificationDataset, TOPIC_TOKEN, PREMISE_TOKEN, CONCLUSION_TOKEN
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import torch
from torch.utils.data import Dataset
import numpy as np
import csv


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
        "f1": f1_score(p.label_ids, preds, average='macro')
    }


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train the model")
    parser.add_argument("--preprocess", action="store_true", help="Whether to preprocess the data or not")
    parser.add_argument("--augment", action="store_true", help="Whether to augment the data or not")
    args = parser.parse_args()

    # Hyperparameters
    EPOCHS = args.epochs
    MAX_LENGTH = 200

    # Set random seeds and device
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # Add special tokens
    tokenizer.add_tokens([TOPIC_TOKEN, PREMISE_TOKEN, CONCLUSION_TOKEN])

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

    test_predictions = []

    # Preprocess data and train models for each task separately
    print("Training models...")
    for task in ["Validity", "Novelty"]:
        print("Task:", task)

        # Load datasets
        print("Loading data...")
        dataset_train = ClassificationDataset("data/TaskA_train.csv",
                                              task=task,
                                              preprocess=args.preprocess,
                                              augment=args.augment)
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
        model.resize_token_embeddings(len(tokenizer))

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
        print(f"F1 score for {task}:", results["eval_f1"])

        # Predict the labels of the test set
        predictions = trainer.predict(transformer_dataset_test)
        predictions = np.argmax(predictions.predictions, axis=1)

        # Save the predictions
        test_predictions.append(predictions)

    # Save the predictions to a csv file with the topic, premise, conclusion and predictions
    print("Saving predictions...")
    # Convert predictions of 0 to -1
    test_predictions[0][test_predictions[0] == 0] = -1
    test_predictions[1][test_predictions[1] == 0] = -1

    dataset_test = ClassificationDataset("data/TaskA_test.csv", task="Validity")

    with open(f"results/predictions_{timestamp}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Topic", "Premise", "Conclusion", "Validity", "Novelty"])
        for i in range(len(dataset_test)):
            writer.writerow(
                [dataset_test.data[i]['Topic'], dataset_test.data[i]['Premise'], dataset_test.data[i]['Conclusion'],
                 test_predictions[0][i], test_predictions[1][i]])







