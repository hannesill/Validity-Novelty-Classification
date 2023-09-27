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
            "Premise": sample["Premise"],
            "Conclusion": paraphrase(sample["Conclusion"]),
            task: sample[task],
            "Confidence": sample["Confidence"]
        }
        new_entries.append(new_entry)

    return new_entries


def create_novel_entries(data, num_new_entries):
    # Choose random other conclusion for each sentence's topic and premise
    novel_entries = []
    for i in range(0, num_new_entries):
        # Choose a random sample
        sample = random.choice(data)
        conclusion = sample["Conclusion"]
        # Choose a random conclusion different from the current conclusion
        while True:
            random_conclusion = random.choice(data)["Conclusion"]
            if random_conclusion != conclusion:
                break

        novel_entry = {
            "Topic": sample["Topic"],
            "Premise": sample["Premise"],
            "Conclusion": random_conclusion,
            "Novelty": sample["Novelty"],
            "Confidence": sample["Confidence"]
        }

        novel_entries.append(novel_entry)

    return novel_entries


class ClassificationDataset(Dataset):
    def __init__(self, file_name, task, augment=False, filter_out="defeasible"):

        # Check if the task is valid
        if task not in ["Validity", "Novelty"]:
            raise ValueError("Invalid task. Task must be either 'Validity' or 'Novelty'.")

        # Check if the filter_out is valid
        if filter_out not in ["defeasible", "majority", "confident"]:
            raise ValueError("Invalid filter_out. filter_out must be either 'defeasible' or 'majority' or 'confident.")
        # Create a list of filters
        filters = []
        for filter_keyword in ["defeasible", "majority", "confident"]:
            filters.append(filter_keyword)
            if filter_keyword == filter_out:
                break
        print("Filters:", filters)

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

        # Filter the data and convert -1 labels to 0
        self.data = []
        for entry in orig_data:
            # Skip the entry if it has a confidence score in the filters
            if entry["Confidence"] in filters:
                continue

            # Convert label
            entry[task] = 0 if entry[task] == -1 else 1

            self.data.append(entry)

            # Balance the data
            # By creating new novel entries
            if task == "Novelty":
                # Calculate the number of novel sentences to create
                num_novel_entries = len([entry for entry in self.data if entry[task] == 1])
                num_new_entries = len(self.data) - num_novel_entries
                novel_entries = create_novel_entries(self.data, num_new_entries)
                self.data += novel_entries

            # By oversampling not valid entries to match the number of valid entries
            elif task == "Validity":
                # Calculate the number of valid and not valid entries
                num_valid_entries = len([entry for entry in self.data if entry["Validity"] == 1])
                num_not_valid_entries = len([entry for entry in self.data if entry["Validity"] == 0])
                num_new_entries = num_valid_entries - num_not_valid_entries

                # Oversample the not valid entries
                not_valid_entries = [entry for entry in self.data if entry["Validity"] == 0]
                new_not_valid_entries = random.choices(not_valid_entries, k=num_new_entries)

                # Add the new entries to the data
                self.data += new_not_valid_entries

        # Augment the data if set to True
        if augment:
            print("Augmenting data...")

            # Further augment the data by paraphrasing the conclusion
            augmenting_factor = 1
            new_data = augment_data(self.data, augmenting_factor, task)
            self.data += new_data

            # Save the augmented data
            augmented_file_name = file_name.replace(".csv", f"_{task}_augmented_{augmenting_factor}.csv")
            df = pd.DataFrame(self.data)
            df.to_csv(augmented_file_name, index=False)

        # Shuffle the data
        random.shuffle(self.data)

        # Get the sentences and labels
        # TODO: Try without special tokens
        self.sentences = [
            f"{TOPIC_TOKEN} {entry['Topic']} {PREMISE_TOKEN} {entry['Premise']} {CONCLUSION_TOKEN} {entry['Conclusion']}".lower()
            for entry in self.data
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