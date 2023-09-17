import torch
from torch.utils.data import Dataset
import pandas as pd
import nltk
from nltk.corpus import wordnet
import random

nltk.download('wordnet')


def get_synonyms(word):
    """Get synonyms of a word."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def synonym_replacement(words, n=1):
    """Replace n words in the sentence with synonyms from WordNet."""
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalpha()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # Only replace up to n words
            break

    return new_words


def augment_data(sentences, labels):
    # Split the sentences and labels by class
    novel_sentences = [sentences[i] for i, lbl in enumerate(labels["novelty"]) if lbl == 1]
    not_novel_sentences = [sentences[i] for i, lbl in enumerate(labels["novelty"]) if lbl == 0]

    difference = len(not_novel_sentences) - len(novel_sentences)

    augmented_sentences = []
    augmented_labels = {
        "validity": [],
        "novelty": []
    }

    # Augment the novel sentences by the difference
    for i in range(difference):
        idx = i % len(novel_sentences)  # Use modulo to wrap around if difference > len(novel_sentences)
        words = novel_sentences[idx].split()
        new_sentence_words = synonym_replacement(words, n=1)
        new_sentence = ' '.join(new_sentence_words)
        augmented_sentences.append(new_sentence)

        # Copy over the labels for augmented sentences
        for label_type, label_values in labels.items():
            augmented_labels[label_type].append(label_values[idx])

    # Combine the original and augmented data
    all_sentences = sentences + augmented_sentences
    all_labels = {
        "validity": labels["validity"] + augmented_labels["validity"],
        "novelty": labels["novelty"] + augmented_labels["novelty"]
    }

    return all_sentences, all_labels


PADDING_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
TOPIC_TOKEN = "<TOPIC>"
PREMISE_TOKEN = "<PREMISE>"
CONCLUSION_TOKEN = "<CONCLUSION>"


class ValidityNoveltyClassificationDataset(Dataset):
    def __init__(self, file_name, augment=False):

        # Get the data from the csv file
        df = pd.read_csv(file_name, encoding="utf8", sep=",")

        # Make the data into a list of tuples
        self.data = [
            {
                "Topic": row["topic"],
                "Premise": row["Premise"],
                "Conclusion": row["Conclusion"],
                "Validity": row["Validity"],
                "Novelty": row["Novelty"]
            }
            for _, row in df.iterrows()
        ]

        # Get the sentences and labels
        self.sentences = []
        validity_labels = []
        novelty_labels = []
        for entry in self.data:
            # Skip the entry if the validity or novelty is 0
            if entry["Validity"] == 0 or entry["Novelty"] == 0:
                continue
            # Concatenate the topic, premise and conclusion with the special tokens
            sentence = f'{TOPIC_TOKEN} {entry["Topic"]} {PREMISE_TOKEN} {entry["Premise"]} {CONCLUSION_TOKEN} {entry["Conclusion"]}'
            self.sentences.append(sentence)

            # Add the labels and replace -1 with 0
            validity_labels.append(0 if entry["Validity"] == -1 else 1)
            novelty_labels.append(0 if entry["Novelty"] == -1 else 1)

        self.labels = {
            "validity": validity_labels,
            "novelty": novelty_labels
        }

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
        return self.sentences[idx], [self.labels["validity"][idx], self.labels["novelty"][idx]]

    def __len__(self):
        return len(self.sentences)


class ValidityNoveltyClassificationCollator:
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
    dataset = ValidityNoveltyClassificationDataset("TaskA_train.csv")
    print(dataset[0])
    print(dataset[1])
    print(len(dataset.vocab))

    collator = ValidityNoveltyClassificationCollator(dataset.vocab)

    # Print longest and shortest topics, premises, and conclusions
    print("Longest")
    print(max([len(topic) for topic in [sample["Topic"] for sample in dataset.data]]))
    print(max([len(premise) for premise in [sample["Premise"] for sample in dataset.data]]))
    print(max([len(conclusion) for conclusion in [sample["Conclusion"] for sample in dataset.data]]))

    print("Shortest")
    print(min([len(topic) for topic in [sample["Topic"] for sample in dataset.data]]))
    print(min([len(premise) for premise in [sample["Premise"] for sample in dataset.data]]))
    print(min([len(conclusion) for conclusion in [sample["Conclusion"] for sample in dataset.data]]))
