# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from collections import defaultdict
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

def fit(epochs, batch_size, criterion1, criterion2, opt, model, train, val, device, scheduler):

    """
    Function to train the neural network
    
    Parameters
    ----------
    epochs: number of epochs
    batch_size: batch size for the training dataloader
    criterion1: loss function for classification outputs- floor, building
    criterion2: loss function for regression outputs- latitude, longitude
    opt: optimizer
    model: model to train the function
    train: training dataset
    val: validation dataset
    device: cpu or gpu
    scheduler: to dynamically update learning rate

    Returns
    -------
    best_model: returns the best model based on validation loss
    """

    train_dl = DataLoader(train, batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size=1, shuffle=True)
    best_valloss = 10**10
    best_model = model
    best_epoch = 0

    lr = defaultdict(list) #to store learning rates
    tloss = defaultdict(list) #to store training loss
    vloss = defaultdict(list) #to store validation loss

    for epoch in range(epochs):

        y_true_train1, y_true_train2, y_true_train3, y_true_train4  = (list() for i in range(4))
        y_pred_train1, y_pred_train2, y_pred_train3, y_pred_train4  = (list() for i in range(4))
        total_loss_train = 0          

        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            
            opt.zero_grad()
            floor, building, latitude, longitude = model(x.float())

            loss = criterion1(floor, y[:,0].long()) + criterion1(building, y[:,1].long()) + criterion2(latitude, y[:,2].unsqueeze(dim=1)) + criterion2(longitude, y[:,3].unsqueeze(dim=1))
            
            loss.backward()
            lr[epoch].append(opt.param_groups[0]['lr'])
            tloss[epoch].append(loss.item())
            opt.step()

            y_true_train1 += list(y[:,0].cpu().data.numpy())
            y_true_train2 += list(y[:,1].cpu().data.numpy())
            y_true_train3 += list(y[:,2].cpu().data.numpy())
            y_true_train4 += list(y[:,3].cpu().data.numpy())

            y_pred_train1 += list(torch.max(floor,1).indices.cpu().data.numpy())
            y_pred_train2 += list(torch.max(building,1).indices.cpu().data.numpy())
            y_pred_train3 += list(latitude.cpu().detach().data.numpy())
            y_pred_train4 += list(longitude.cpu().detach().data.numpy())

            total_loss_train += loss.item()

        train_floor_acc = accuracy_score(y_true_train1, y_pred_train1)
        train_building_acc = accuracy_score(y_true_train2, y_pred_train2)
        train_lat_rmse = mean_squared_error(y_true_train3, y_pred_train3, squared=False)
        train_long_rmse = mean_squared_error(y_true_train4, y_pred_train4, squared=False)

        train_loss = total_loss_train/len(train_dl)
        
        if val_dl:

            y_true_val1, y_true_val2, y_true_val3, y_true_val4  = (list() for i in range(4))
            y_pred_val1, y_pred_val2, y_pred_val3, y_pred_val4  = (list() for i in range(4))

            total_loss_val = 0

            for x, y in val_dl:

                x = x.to(device)
                y = y.to(device).float()

                floor, building, latitude, longitude = model(x)
                loss = criterion1(floor, y[:,0].long()) + criterion1(building, y[:,1].long()) + criterion2(latitude, y[:,2].unsqueeze(dim=1)) + criterion2(longitude, y[:,3].unsqueeze(dim=1))
                
                y_true_val1 += list(y[:,0].cpu().data.numpy())
                y_true_val2 += list(y[:,1].cpu().data.numpy())
                y_true_val3 += list(y[:,2].cpu().data.numpy())
                y_true_val4 += list(y[:,3].cpu().data.numpy())

                y_pred_val1 += list(torch.max(floor,1).indices.cpu().data.numpy())
                y_pred_val2 += list(torch.max(building,1).indices.cpu().data.numpy())
                y_pred_val3 += list(latitude.cpu().detach().data.numpy())
                y_pred_val4 += list(longitude.cpu().detach().data.numpy())

                total_loss_val += loss.item()
                vloss[epoch].append(loss.item())

            val_floor_acc = accuracy_score(y_true_val1, y_pred_val1)
            val_building_acc = accuracy_score(y_true_val2, y_pred_val2)
            val_lat_rmse = mean_squared_error(y_true_val3, y_pred_val3, squared=False)
            val_long_rmse = mean_squared_error(y_true_val4, y_pred_val4, squared=False)

            valloss = total_loss_val/len(val)
            scheduler.step(valloss)

            if valloss < best_valloss:
                best_valloss = valloss
                best_model = model
                #best_model_path = 'trained_model-epoch{}.pth'.format(epoch)
                #best_epoch = epoch
                #torch.save(model.state_dict(), best_model_path)
            
            print(f'Epoch {epoch}:')
            print(f'train_loss: {train_loss:.4f} train_floor_acc: {train_floor_acc:.4f} train_building_acc: {train_building_acc:.4f} train_latitude_rmse: {train_lat_rmse:.2f} train_longitude_rmse: {train_long_rmse:.2f}')
            print(f'val_loss: {valloss:.4f} val_floor_acc: {val_floor_acc:.4f} val_building_acc: {val_building_acc:.4f} val_latitude_rmse: {val_lat_rmse:.2f} val_longitude_rmse: {val_long_rmse:.2f}')
    
    return best_model
