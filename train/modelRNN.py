import torch.optim as optim
# from train.model import RNNClassifier
import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    
    Args:
            vocab_size: vocab size
            embedding_size: embedding size
            rnn_model:  LSTM or GRU
            embedding_tensor:
            padding_index:
            hidden_size: hidden size of rnn module
            num_layers:  number of layers in rnn module
            batch_first: batch first option
            
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, rnn_model='LSTM', padding_index=0, num_layers=1):
        """
        Initialize the model by settingg up the various layers.
        """
        super(RNNClassifier, self).__init__()
  
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index)
        
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers)
            
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers)
            
        else:
            raise LookupError('Only support LSTM and GRU')
            
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        rnn_out, _ = self.rnn(embeds)
        out = self.dense(rnn_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())



