import random
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


PADDING_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
TOPIC_TOKEN = "<TOPIC>"
PREMISE_TOKEN = "<PREMISE>"
CONCLUSION_TOKEN = "<CONCLUSION>"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)


def paraphrase(
        question,
        num_beams=2,
        num_beam_groups=2,
        num_return_sequences=1,
        repetition_penalty=5.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        max_length=500
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)

    outputs = model.generate(
        input_ids, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    ).cpu()

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


def augment_data(data, augmenting_factor, task):
    new_entries = []
    num_new_entries = int(len(data) * augmenting_factor)
    for _ in tqdm(range(0, num_new_entries)):
        # Select random sample and paraphrase the conclusion
        sample = random.choice(data)
        new_entry = {
            "Topic": sample["Topic"],
            "Premise": paraphrase(sample["Premise"]),
            "Conclusion": paraphrase(sample["Conclusion"]),
            task: sample[task],
            "Confidence": sample["Confidence"]
        }
        new_entries.append(new_entry)

    return new_entries


class ClassificationDataset(Dataset):
    def __init__(self, file_name, task, preprocess=False, augment=False):

        # Check if the task is valid
        if task not in ["Validity", "Novelty"]:
            raise ValueError("Invalid task. Task must be either 'Validity' or 'Novelty'.")

        # Get the data from the csv file
        df = pd.read_csv(file_name, encoding="utf8", sep=",")

        # Make the data into a list of tuples
        orig_data = [
            {
                "Topic": row["topic"],
                "Premise": row["Premise"],
                "Conclusion": row["Conclusion"],
                task: row[task],
                "Confidence": row[f"{task}-Confidence"]
            }
            for _, row in df.iterrows()
        ]

        # Filter out defeasible samples and convert -1 labels to 0
        self.data = []
        for entry in orig_data:
            # Filter out defeasible samples
            if entry[task] == 0:
                continue
            # Convert label
            entry[task] = 0 if entry[task] == -1 else 1

            self.data.append(entry)

        if preprocess:
            # Filter out samples with confidence label of "majority"
            self.data = [entry for entry in self.data if entry["Confidence"] != "majority"]
            # Oversample not novel entries
            if task == "Novelty":
                # Calculate the number of novel sentences to create
                num_not_novel_entries = len([entry for entry in self.data if entry[task] == 0])
                num_novel_entries = len([entry for entry in self.data if entry[task] == 1])
                num_new_entries = num_not_novel_entries - num_novel_entries

                # Oversample the novel entries
                novel_entries = [entry for entry in self.data if entry[task] == 1]
                new_novel_entries = random.choices(novel_entries, k=num_new_entries)

                # Add the new entries to the data
                self.data += new_novel_entries

            # Oversample the invalid entries
            elif task == "Validity":
                # Calculate the number of invalid sentences to create
                num_valid_entries = len([entry for entry in self.data if entry[task] == 1])
                num_invalid_entries = len([entry for entry in self.data if entry[task] == 0])
                num_new_entries = num_valid_entries - num_invalid_entries

                # Oversample the invalid entries
                invalid_entries = [entry for entry in self.data if entry[task] == 0]
                new_invalid_entries = random.choices(invalid_entries, k=num_new_entries)

                # Add the new entries to the data
                self.data += new_invalid_entries

            # Shuffle the data
            random.shuffle(self.data)

        # Get the sentences and labels
        self.sentences = [
            f"{TOPIC_TOKEN} {entry['Topic']} {PREMISE_TOKEN} {entry['Premise']} {CONCLUSION_TOKEN} {entry['Conclusion']}".lower()
            for entry in self.data
        ]
        # Remove special symbols from the sentences
        self.sentences = [
            sentence.replace("–", " ").replace("—", " ").replace("“", "").replace("”", "").replace("á", "a")
            for sentence in self.sentences
        ]

        self.labels = [entry[task] for entry in self.data]

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
    print("Device:", device)

    text = "This is a test sentence, which was written in the north of Bielefeld, a mid-sized city in the north-west of Germany."
    print(paraphrase(text))

    dataset = ClassificationDataset("data/TaskA_train.csv", "Validity", augment=False)
    print(dataset[0])
    print(dataset.data[0])
    print(paraphrase(dataset.data[0]["Premise"]))
    print(paraphrase(dataset.data[0]["Conclusion"]))
    print(len(dataset.vocab))

    # Print longest premise and conclusion
    print(max([len(entry["Premise"]) for entry in dataset.data]))
    print(max([len(entry["Conclusion"]) for entry in dataset.data]))