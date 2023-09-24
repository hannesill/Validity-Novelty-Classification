import torch
from torch.utils.data import Dataset
import pandas as pd


# TODO: Implement data augmentation
def augment_data(sentences, labels):
    return sentences, labels


PADDING_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
TOPIC_TOKEN = "<TOPIC>"
PREMISE_TOKEN = "<PREMISE>"
CONCLUSION_TOKEN = "<CONCLUSION>"


class ClassificationDataset(Dataset):
    def __init__(self, file_name, task, augment=False):

        if task not in ["Validity", "Novelty"]:
            raise ValueError("Invalid task. Task must be either 'Validity' or 'Novelty'.")

        self.task = task

        # Get the data from the csv file
        df = pd.read_csv(file_name, encoding="utf8", sep=",")

        # Make the data into a list of tuples
        self.data = [
            {
                "Topic": row["topic"],
                "Premise": row["Premise"],
                "Conclusion": row["Conclusion"],
                task: row[task],
                "Confidence": row[f"{task}-Confidence"]
            }
            for _, row in df.iterrows()
        ]

        # Get the sentences and labels
        self.sentences = []
        self.labels = []

        for entry in self.data:
            # Skip the entry if the validity and novelty is 0
            # TODO: Try out ignoring all entries with no high confidence
            if entry["Confidence"] == "defeasible" or entry["Confidence"] == "majority" or entry["Confidence"] == "confident":
                continue

            # Concatenate the topic, premise and conclusion with the special tokens
            sentence = f'{TOPIC_TOKEN} {entry["Topic"]} {PREMISE_TOKEN} {entry["Premise"]} {CONCLUSION_TOKEN} {entry["Conclusion"]}'

            # Add the sentence with its corresponding label to the respective lists
            # Replace -1 with 0 for the labels
            if entry[task] == 1 or entry[task] == -1:
                self.sentences.append(sentence)
                self.labels.append(0 if entry[task] == -1 else 1)

        # If augment is True augment the data
        if augment:
            self.sentences, self.labels = augment_data(self.sentences, self.labels)

        # Get the vocab
        self.vocab = self.get_vocab()

    def get_vocab(self):
        # Initialize the vocab with the special tokens
        vocab = {
            PADDING_TOKEN: 0,
            UNKNOWN_TOKEN: 1,
            TOPIC_TOKEN: 2,
            PREMISE_TOKEN: 3,
            CONCLUSION_TOKEN: 4
        }

        # Add the words of the sentences to the vocab
        for sentence in self.sentences:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = len(vocab)

        return vocab

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    def __len__(self):
        return len(self.sentences)


class ClassificationCollator:
    def __init__(self, vocab, max_length=500):
        self.vocab = vocab
        self.max_length = max_length

    def tokenize_and_pad(self, sequence):
        tokens = [self.vocab[word] if word in self.vocab else self.vocab[UNKNOWN_TOKEN] for word in sequence.split()]
        padded_tokens = tokens + [self.vocab[PADDING_TOKEN]] * (self.max_length - len(tokens))
        return padded_tokens

    def __call__(self, batch):
        # Get the sentences
        sentences = []
        for sentence, _ in batch:
            # Truncate the sentence if it is too long
            sentence = sentence[:self.max_length]
            # Tokenize and pad the sentence
            sentence = self.tokenize_and_pad(sentence)

            # Add the sentence and label to the lists
            sentences.append(sentence)

        # Get the labels
        labels = [label for _, label in batch]

        return {
            "sentences": torch.tensor(sentences, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32)
        }


if __name__ == "__main__":
    dataset = ClassificationDataset("data/TaskA_train.csv", "Validity", augment=False)
    print(dataset[0])
    print(dataset[1])
    print(len(dataset.vocab))

    collator = ClassificationCollator(dataset.vocab)

    # Print longest and shortest topics, premises, and conclusions
    print("Longest")
    print(max([len(topic) for topic in [sample["Topic"] for sample in dataset.data]]))
    print(max([len(premise) for premise in [sample["Premise"] for sample in dataset.data]]))
    print(max([len(conclusion) for conclusion in [sample["Conclusion"] for sample in dataset.data]]))

    print("Shortest")
    print(min([len(topic) for topic in [sample["Topic"] for sample in dataset.data]]))
    print(min([len(premise) for premise in [sample["Premise"] for sample in dataset.data]]))
    print(min([len(conclusion) for conclusion in [sample["Conclusion"] for sample in dataset.data]]))
