
import numpy as np
import torch
import matplotlib.pyplot as plt

def find_acc(y_pred, y):
  y_pred = (y_pred.view(-1).detach().cpu().numpy()>0.5).astype('float')
  y = y.view(-1).detach().cpu().numpy().astype('float')  
  acc = np.mean((y== y_pred).astype('float'))
  return acc

def train(model, criterion, opt, train_loader, val_loader, n_epochs, device, show_epoch, save_epoch):
  loss_train, loss_val=[],[]
  acc_train, acc_val=[], []
  for epoch in range(n_epochs):
    ############################## TRAIN !!!
    model.train()
    loss_t, acc_t=[], []
    for i, (x, y) in enumerate(train_loader):
      x= x.to(device)
      y= y.to(device)
      opt.zero_grad()
      
      y_pred = model(x)

      loss= criterion(y_pred, y)
      loss.backward()
      opt.step()

      loss_t.append(loss.item())
      acc_t.append(find_acc(y_pred, y))

    loss_train.append(np.mean(loss_t))
    acc_train.append(np.mean(acc_t))

    ################################ TEST !!!
    model.eval()
    loss_t, acc_t=[],[]
    for i, (x,y) in enumerate(val_loader):
      x= x.to(device)
      y= y.to(device)

      with torch.no_grad():
        y_pred = model(x)
        loss= criterion(y_pred, y)
        loss_t.append(loss.item())
        acc_t.append(find_acc(y_pred, y))

    loss_val.append(np.mean(loss_t))
    acc_val.append(np.mean(acc_t))


    if epoch%show_epoch==0:
      plt.figure(figsize= (9, 2.5))
      plt.subplot(1,2,1)
      plt.plot(loss_train, label='train')
      plt.plot(loss_val, label= 'val')
      plt.legend()
      plt.title(f'loss : after {epoch} epochs')

      plt.subplot(1,2,2)
      plt.plot(acc_train, label='train')
      plt.plot(acc_val, label= 'val')
      plt.legend()
      plt.title(f'accuracy : after {epoch} epochs')

      plt.show()

    if epoch% save_epoch==0:
      print('skipping saving...')