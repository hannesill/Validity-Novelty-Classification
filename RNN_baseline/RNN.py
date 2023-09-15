import torch
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
