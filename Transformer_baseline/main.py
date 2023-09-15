from data.dataset import ValidityNoveltyClassificationDataset, ValidityNoveltyClassificationCollator
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader


class TransformerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # Hyperparameters
    MAX_LENGTH = 200
    EPOCHS = 3

    dataset_train = ValidityNoveltyClassificationDataset("../data/TaskA_train.csv")
    dataset_valid = ValidityNoveltyClassificationDataset("../data/TaskA_dev.csv")

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
    for task in ["validity"]: # TODO: Add "novelty" later
        # Create the model
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch"
        )
        model = RobertaForSequenceClassification.from_pretrained("roberta-base")

        # Convert the labels to tensors
        labels_train = torch.tensor(dataset_train.labels[task])
        labels_valid = torch.tensor(dataset_valid.labels[task])

        # Create the datasets
        transformer_dataset_train = TransformerDataset(encodings_train, labels_train)
        transformer_dataset_valid = TransformerDataset(encodings_valid, labels_valid)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=transformer_dataset_train,
            eval_dataset=transformer_dataset_valid
        )

        trainer.train()