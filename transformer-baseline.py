import datetime
from sklearn.metrics import f1_score, accuracy_score
from data.dataset import ValidityNoveltyClassificationDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import torch
from torch.utils.data import Dataset
import numpy as np


class TransformerDataset(Dataset):
    def __init__(self, encodings, labels):
        # Transform the encodings and labels to tensors
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
        "f1": f1_score(p.label_ids, preds, average='weighted')
    }


if __name__ == "__main__":
    # Set random seeds and device
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")

    # Hyperparameters
    MAX_LENGTH = 200
    EPOCHS = 3

    dataset_train = ValidityNoveltyClassificationDataset("data/TaskA_train.csv")
    dataset_valid = ValidityNoveltyClassificationDataset("data/TaskA_dev.csv")

    # Preprocess the data
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Encode the sentences
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

    # Train two models, one for each task
    for task in ["validity", "novelty"]:
        # Create the model
        training_args = TrainingArguments(
            output_dir="results",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch"
        )
        model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)

        # Convert the labels to tensors
        labels_train = torch.tensor(dataset_train.labels[task])
        labels_valid = torch.tensor(dataset_valid.labels[task])

        # Create the datasets
        transformer_dataset_train = TransformerDataset(encodings_train, labels_train)
        transformer_dataset_valid = TransformerDataset(encodings_valid, labels_valid)

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
        dataset_test = ValidityNoveltyClassificationDataset("data/TaskA_test.csv")

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
        transformer_dataset_test = TransformerDataset(encodings_test, torch.tensor(dataset_test.labels[task]))

        # Calculate and print the accuracy and F1 score
        results = trainer.evaluate(transformer_dataset_test)
        print(f"Accuracy for {task}:", results["eval_accuracy"])
        print(f"F1 score for {task}:", results["eval_f1"])

