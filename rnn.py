from data.dataset import ValidityNoveltyClassificationDataset, ValidityNoveltyClassificationCollator
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


class RNN_Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx, num_layers=8, output_dim=2):
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

        return torch.sigmoid(output)



def calc_f1_score(preds, labels):
    # Calculate the number of true positives, false positives and false negatives
    TP = ((preds == 1) & (labels == 1)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    # Calculate precision, recall and F1 score
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return f1_score


def evaluate_model(model, dataloader_dev):
    accuracies = []
    f1_scores = []

    for batch in dataloader_dev:
        sentences = batch["sentences"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(sentences)

        # Separate out the predictions
        validity_preds = outputs[:, 0]
        novelty_preds = outputs[:, 1]

        # Binarize the predictions
        validity_preds = (validity_preds > 0.5).long()
        novelty_preds = (novelty_preds > 0.5).long()

        # Separate out the labels
        validity_labels = labels[:, 0]
        novelty_labels = labels[:, 1]

        # Compute accuracy
        batch_validity_accuracy = (validity_preds == validity_labels).sum().item() / len(validity_preds)
        batch_novelty_accuracy = (novelty_preds == novelty_labels).sum().item() / len(novelty_preds)

        # Compute F1 Scores
        batch_f1_validity = calc_f1_score(validity_preds, validity_labels)
        batch_f1_novelty = calc_f1_score(novelty_preds, novelty_labels)

        # Append to list
        accuracies.append((batch_validity_accuracy, batch_novelty_accuracy))
        f1_scores.append((batch_f1_validity, batch_f1_novelty))

    # Compute overall accuracies and F1 scores
    validity_accuracy = sum([x[0] for x in accuracies]) / len(accuracies)
    novelty_accuracy = sum([x[1] for x in accuracies]) / len(accuracies)
    validity_f1 = sum([x[0] for x in f1_scores]) / len(f1_scores)
    novelty_f1 = sum([x[1] for x in f1_scores]) / len(f1_scores)
    overall_accuracy = sum([x[0] + x[1] for x in accuracies]) / 2 * len(accuracies)
    overall_f1 = sum([x[0] + x[1] for x in f1_scores]) / 2 * len(f1_scores)

    # Print Results
    print(f"Accuracy for Validity: {validity_accuracy:.4f}")
    print(f"Accuracy for Novelty: {novelty_accuracy:.4f}")
    print(f"F1 score for Validity: {validity_f1:.4f}")
    print(f"F1 score for Novelty: {novelty_f1:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall F1 score: {overall_f1:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Load dataset
    print("Loading dataset...")
    dataset_train = ValidityNoveltyClassificationDataset("data/TaskA_train.csv")
    dataset_dev = ValidityNoveltyClassificationDataset("data/TaskA_dev.csv")
    dataset_test = ValidityNoveltyClassificationDataset("data/TaskA_test.csv")

    # Hyperparameters
    batch_size = 64
    emb_dim = 64
    hidden_dim = 64
    num_layers = 3
    num_epochs = 3
    eval_every = 1
    learning_rate = 0.001
    pad_idx = dataset_train.vocab["<PAD>"]

    # Create dataloader
    collator = ValidityNoveltyClassificationCollator(dataset_train.vocab)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collator)

    # Create model
    model = RNN_Model(len(dataset_train.vocab), emb_dim, hidden_dim, pad_idx, num_layers=num_layers).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create loss function
    criterion = torch.nn.BCELoss()

    # Train model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Iterate over each batch
        for batch in dataloader_train:
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

    print("Training complete! =======================> Saving model...")
    # torch.save(model.state_dict(), "model.pt")

    # Test model
    print("Testing model...")
    model.eval()
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collator)
    evaluate_model(model, dataloader_test)
