import torch
import torch.nn as nn

class HypernymHyponymLSTM(nn.Module):
    def __init__(self, num_hyponyms, num_hypernyms, embedding_dim, hidden_dim):
        super(HypernymHyponymLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hyponym_embedding = nn.Embedding(num_hyponyms, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_hypernyms)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hyponym):
        hyponym_embedded = self.hyponym_embedding(hyponym)
        lstm_out1, _ = self.lstm1(hyponym_embedded)
        lstm_out2, _ = self.lstm2(lstm_out1)
        linear_out = self.linear(lstm_out2)
        return linear_out

class HypernymHyponymGRU(nn.Module):
    def __init__(self, num_hyponyms, num_hypernyms, embedding_dim, hidden_dim):
        super(HypernymHyponymGRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hyponym_embedding = nn.Embedding(num_hyponyms, embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, num_layers=2)  # Two GRU layers
        self.gru2 = nn.GRU(hidden_dim, hidden_dim)  # Second GRU layer
        self.linear = nn.Linear(hidden_dim, num_hypernyms)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hyponym):
        hyponym_embedded = self.hyponym_embedding(hyponym)
        gru1_out, _ = self.gru1(hyponym_embedded)
        gru2_out, _ = self.gru2(gru1_out)  # Pass output of first GRU layer to second GRU layer
        linear_out = self.linear(gru2_out)
        return linear_out

class HypernymHyponymFC(nn.Module):
    def __init__(self, num_hyponyms, num_hypernyms, embedding_dim, hidden_dim):
        super(HypernymHyponymFC, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hyponym_embedding = nn.Embedding(num_hyponyms, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_hypernyms)
        self.dropout = nn.Dropout(0.5)

    def forward(self, hyponym):
        hyponym_embedded = self.hyponym_embedding(hyponym)
        linear1_out = self.dropout(self.linear1(hyponym_embedded))
        linear2_out = self.linear2(linear1_out)
        return linear2_out
