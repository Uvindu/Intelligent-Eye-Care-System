from torch.nn import functional as F
from torch import nn
import torch



class conv_block(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size= 3, stride=2, dropout_ratio= 0.3):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size= kernel_size, stride= stride, padding= 1),
            nn.MaxPool2d(kernel_size= 2),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.BatchNorm2d(out_channels))
    def forward(self, x): # x.shape: (m, 1, n, n)
        return self.block(x)



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


# CNN: (32, 1, eye_size, eye_size) -> (32, seq_len, 1) ## feature_len= 1 

class cnn_lstm(nn.Module): 
  def __init__(self, lstm_input_feature_size= 10, lstm_hidden_size= 20, n_features_to_linear= 4, num_lstm_layers= 3, bias=False, dropout=0.3, bidirectional=False):
    super(cnn_lstm, self).__init__()
    self.lstm_input_feature_size = lstm_input_feature_size

    self.eye_preprocess_net = nn.Sequential(
        conv_block(1, 32, stride=2, dropout_ratio=  dropout),
        conv_block(32, 128, stride=2, dropout_ratio=  dropout), 
        conv_block(128, self.lstm_input_feature_size, stride=1, dropout_ratio=  dropout), 
        )

    batch_first= True
    self.n_features_to_linear = n_features_to_linear

    self.bidirectional= bidirectional
    self.lstm_hidden_size= lstm_hidden_size

    self.lstm1 = nn.LSTM(self.lstm_input_feature_size, self.lstm_hidden_size, num_lstm_layers, bias  , batch_first , dropout, self.bidirectional)
    self.lstm2 = nn.LSTM(self.lstm_hidden_size       , 1                    , 1              , False , batch_first , 0      , self.bidirectional)

    self.sequential_block= nn.Sequential(
        nn.Linear(self.n_features_to_linear, 1),
        nn.Sigmoid())

  def forward(self, x):  # x.shape-> (batch_size, sequence_length, eye_size, eye_size)
    n_samples= x.shape[0]
    sequence_length= x.shape[1]

    x= x.view(n_samples*sequence_length, 1, x.shape[2], x.shape[3]) # x.shape-> (n_samples*sequence_length, 1, eye_size, eye_size)
    x = F.interpolate(x, size=(32, 32), mode='bicubic', align_corners=False)  # x.shape-> (n_samples*sequence_length, 1, 32, 32)

    x = self.eye_preprocess_net(x) # (n_samples*sequence_length, self.lstm_input_feature_size, 1, 1)

    x= x.view(n_samples, sequence_length, self.lstm_input_feature_size)

    #x.shape -> (batch_size, sequence_length, feature_length)    
    x, _ = self.lstm1(x)
    x= x.view(n_samples, sequence_length, self.lstm_hidden_size, -1)
    x = torch.sum(x, dim= -1)
    
    x, _ = self.lstm2(x)
    x= x.view(n_samples, sequence_length, 1, -1)
    x = torch.sum(x, dim= -1)

    x= x.view(n_samples, sequence_length)[:, -self.n_features_to_linear:]
    x= self.sequential_block(x)

    return x.view(-1)

