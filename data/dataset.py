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
print("Device:", device)

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


def augment_data(data, num_new_sentences, task):
    new_sentences = []
    new_labels = []

    for i in tqdm(range(0, num_new_sentences)):
        # Select random sample
        sample = random.choice(data)
        topic = sample["Topic"]
        premise = sample["Premise"]
        conclusion = sample["Conclusion"]
        # Create the new sentence by paraphrasing the conclusion
        new_sentence = f'{TOPIC_TOKEN} {topic} {PREMISE_TOKEN} {premise} {CONCLUSION_TOKEN} {paraphrase(conclusion)[0]}'.lower()
        # Add the new sentence and label to the lists
        new_sentences.append(new_sentence)
        new_labels.append(sample[task])

    return new_sentences, new_labels


def create_novel_sentences(data, num_new_sentences):
    # Choose random other conclusion for each sentence's topic and premise
    novel_sentences = []
    for i in range(0, num_new_sentences):
        # Choose a random sample
        sample = random.choice(data)
        topic = sample["Topic"]
        premise = sample["Premise"]
        conclusion = sample["Conclusion"]
        # Choose a random conclusion different from the current conclusion
        while True:
            random_conclusion = random.choice(data)["Conclusion"]
            if random_conclusion != conclusion:
                break
        # Create the novel sentence
        novel_sentences.append(f'{TOPIC_TOKEN} {topic} {PREMISE_TOKEN} {premise} {CONCLUSION_TOKEN} {random_conclusion}'.lower())

    return novel_sentences


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
            # Skip the entry if it has a confidence score in the filters
            if entry["Confidence"] in filters:
                continue

            # Concatenate the topic, premise and conclusion with the special tokens and lower case the sentence
            sentence = f'{TOPIC_TOKEN} {entry["Topic"]} {PREMISE_TOKEN} {entry["Premise"]} {CONCLUSION_TOKEN} {entry["Conclusion"]}'.lower()

            # Add the sentence with its corresponding label to the respective lists
            # Replace -1 with 0 for the labels
            if entry[task] == 1 or entry[task] == -1:
                self.sentences.append(sentence)
                self.labels.append(0 if entry[task] == -1 else 1)

        # If augment is True and task is Novelty, create novel sentences
        if augment:
            if task == "Novelty":
                # Calculate the number of novel sentences to create
                num_novel_sentences = len([entry for entry in self.labels if entry == 1])
                num_new_sentences = len(self.sentences) - num_novel_sentences
                novel_sentences = create_novel_sentences(self.data, num_new_sentences)
                self.sentences += novel_sentences
                self.labels += [1] * len(novel_sentences)

            # Augment the data by paraphrasing the conclusion
            print("Augmenting data...")
            new_sentences, new_labels = augment_data(self.data, len(self.sentences), task)
            self.sentences += new_sentences
            self.labels += new_labels

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


if __name__ == "__main__":#

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