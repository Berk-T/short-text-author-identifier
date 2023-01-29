import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence


class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_classes, num_layers=1, **additional_kwargs):

        super().__init__()

        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'num_classes': num_classes,
            'num_layers': num_layers,
            **additional_kwargs
        }

        self.hparams = hparams

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, 0)

        self.NN = nn.LSTM(embedding_dim, hidden_size,
                          num_layers=num_layers)

        #self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.Sigmoid(), nn.Linear(
            64, 16), nn.Sigmoid(), nn.Linear(16, num_classes))

    def forward(self, sequence, lengths=None):

        output = None

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        seq_len, batch_size = sequence.size()
        hidden = self.init_hidden(batch_size)
        embeds = self.embedding(sequence)

        if lengths is None:
            lengths = torch.full((batch_size,), fill_value=seq_len).to(device)

        packed = pack_padded_sequence(
            embeds, lengths.to('cpu'), enforce_sorted=False)

        out, hidden = self.NN(packed, hidden)
        hidden = hidden[-1].permute(1, 0, 2)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        output = output.view(batch_size, 10)
        return output

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return (torch.zeros(self.hparams['num_layers'], batch_size, self.hparams['hidden_size']).to(device), torch.zeros(self.hparams['num_layers'], batch_size, self.hparams['hidden_size']).to(device))

    def predict(self, sequence, length=None):
        output = self.forward(sequence, length)
        pred = torch.argmax(output)
        return pred
