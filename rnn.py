import csv
import random
from datetime import datetime

import numpy as np
from data.dataset import ClassificationDataset, ClassificationCollator
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score


class RNN_Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx, num_layers=8, output_dim=1):
        super(RNN_Model, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Embedding layer for input
        self.embedding = nn.Embedding(vocab_size, emb_dim, pad_idx)

        # LSTM cell for RNN
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, dropout=0.2)

        # Fully connected linear layer for classification
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sentence):
        # Embed the sequences
        embedded_sentence = self.embedding(sentence.transpose(0, 1))

        # Process each with the RNN
        _, (hidden_state, _) = self.rnn(embedded_sentence)
        final_hidden_state = hidden_state[-1]

        # Pass the final hidden state through the linear layer for classification
        output = self.fc(final_hidden_state.squeeze(0))

        return torch.sigmoid(output.squeeze(1))


def evaluate_model(model, dataloader_dev):
    f1_scores = []

    for batch in dataloader_dev:
        sentences = batch["sentences"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(sentences)

        # Binarize the predictions
        preds = (outputs > 0.5).long()

        # Compute f1
        batch_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

        # Append to list
        f1_scores.append(batch_f1)

    # Compute overall accuracies and F1 scores
    overall_f1 = sum(f1_scores) / len(f1_scores)

    # Print Results
    print(f"F1 score: {overall_f1:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Hyperparameters
    batch_size = 64
    emb_dim = 64
    hidden_dim = 64
    num_layers = 3
    num_epochs = 3
    eval_every = 1
    learning_rate = 0.01

    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create loss function
    criterion = torch.nn.BCELoss()

    test_predictions = []

    # Train model for each task
    print("Training model...")
    for task in ["Validity", "Novelty"]:
        print(f"Task: {task}")

        # Load dataset
        print("Loading dataset...")
        dataset_train = ClassificationDataset("data/TaskA_train.csv", task=task)
        dataset_dev = ClassificationDataset("data/TaskA_dev.csv", task=task)
        dataset_test = ClassificationDataset("data/TaskA_test.csv", task=task)
        pad_idx = dataset_train.vocab["<PAD>"]

        # Create dataloader
        collator = ClassificationCollator(dataset_train.vocab)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collator)

        # Create model
        model = RNN_Model(len(dataset_train.vocab), emb_dim, hidden_dim, pad_idx, num_layers=num_layers).to(device)

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train model
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1} / {num_epochs} =======================> Training...")
            model.train()
            total_loss = 0

            # Iterate over each batch
            for batch in tqdm(dataloader_train):
                optimizer.zero_grad()

                # Unpack the batch and send to device
                sentences = batch["sentences"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(sentences)

                # Compute loss
                loss = criterion(outputs, labels)
                loss.backward()
                total_loss += loss.item()

                optimizer.step()

            print(f"Epoch {epoch + 1} / {num_epochs} =======================> loss: {total_loss:.4f}")

            if (epoch + 1) % eval_every == 0:
                # Validate model
                print("Evaluating model...")
                model.eval()
                dataloader_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=collator)
                evaluate_model(model, dataloader_dev)

        # Test model
        print("Testing model...")
        model.eval()
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collator)
        evaluate_model(model, dataloader_test)

        # Predict the labels of the test set
        print("Predicting labels...")
        preds = []
        for batch in dataloader_test:
            sentences = batch["sentences"].to(device)
            outputs = model(sentences)
            preds.append((outputs > 0.5).long().cpu().numpy())

        test_predictions.append(np.concatenate(preds))

    # Save the predictions to a csv file with the topic, premise, conclusion and predictions
    print("Saving predictions...")
    # Convert predictions of 0 to -1
    test_predictions[0][test_predictions[0] == 0] = -1
    test_predictions[1][test_predictions[1] == 0] = -1

    dataset_test = ClassificationDataset("data/TaskA_test.csv", task="Validity")

    with open(f"results/predictions_{timestamp}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "Premise", "Conclusion", "predicted validity", "predicted novelty"])
        for i in range(len(dataset_test)):
            writer.writerow(
                [dataset_test.data[i]['Topic'], dataset_test.data[i]['Premise'], dataset_test.data[i]['Conclusion'],
                 test_predictions[0][i], test_predictions[1][i]])
