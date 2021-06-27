
from torch import nn
import torch

class simple_lstm(nn.Module): 
  def __init__(self, lstm_input_feature_size= 10, lstm_hidden_size= 20, n_features_to_linear= 4, num_lstm_layers= 3, bias=False, dropout=0.3, bidirectional=False):
    super(simple_lstm, self).__init__()

    batch_first= True
    
    self.n_features_to_linear = n_features_to_linear
    self.lstm_input_feature_size = lstm_input_feature_size

    self.bidirectional= bidirectional
    self.lstm_hidden_size= lstm_hidden_size

    self.lstm1 = nn.LSTM(self.lstm_input_feature_size, self.lstm_hidden_size, num_lstm_layers, bias  , batch_first , dropout, self.bidirectional)
    self.lstm2 = nn.LSTM(self.lstm_hidden_size       , 1                    , 1              , False , batch_first , 0      , self.bidirectional)

    self.sequential_block= nn.Sequential(
        nn.Linear(self.n_features_to_linear, 1),
        nn.Sigmoid())

  def forward(self, x): #x.shape -> (batch_size, sequence_length, feature_length)
    n_samples= x.shape[0]
    sequence_length= x.shape[1]

    x = x.view(n_samples, sequence_length, self.lstm_input_feature_size)
    
    x, _ = self.lstm1(x)
    x= x.view(n_samples, sequence_length, self.lstm_hidden_size, -1)
    x = torch.sum(x, dim= -1)
    
    x, _ = self.lstm2(x)
    x= x.view(n_samples, sequence_length, 1, -1)
    x = torch.sum(x, dim= -1)

    x= x.view(n_samples, sequence_length)[:, -self.n_features_to_linear:]
    x= self.sequential_block(x)

    return x.view(-1)