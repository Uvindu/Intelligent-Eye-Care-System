import torch

class get_dataset(torch.utils.data.Dataset):
  def __init__(self, dataset):
    super(get_dataset, self)
    self.dataset= dataset
  
  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    data= self.dataset[idx]
    features= torch.tensor(data[0])[0].view(-1, 1).float()
    return features, torch.tensor(data[1]).float()