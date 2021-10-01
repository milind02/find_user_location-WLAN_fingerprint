import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import pandas as pd

def accuracy(dataset, model, scale=None):

    """
    Find accuracy and rmse of the four outputs and prints them
    
    Parameters
    ----------
    dataset: dataset to test the accuracy of the model
    model: model used to find the accuracy

    """

    dist_sum = 0
    correct_floor = 0
    correct_building = 0

    with torch.no_grad():

        dataloader = DataLoader(dataset)

        y_true1, y_true2, y_true3, y_true4  = (list() for i in range(4))
        y_pred1, y_pred2, y_pred3, y_pred4  = (list() for i in range(4))

        for x, y in dataloader:

            floor, building, latitude, longitude = model(x.float())

            y_true1 += list(y[:,0].cpu().data.numpy())
            y_true2 += list(y[:,1].cpu().data.numpy())
            y_true3 += list(y[:,2].cpu().data.numpy())
            y_true4 += list(y[:,3].cpu().data.numpy())

            y_pred1 += list(torch.max(floor,1).indices.cpu().data.numpy())
            y_pred2 += list(torch.max(building,1).indices.cpu().data.numpy())
            y_pred3 += list(latitude.cpu().detach().data.numpy()[0])
            y_pred4 += list(longitude.cpu().detach().data.numpy()[0])

        floor_acc = accuracy_score(y_true1, y_pred1)
        building_acc = accuracy_score(y_true2, y_pred2)
        true = scale.inverse_transform(pd.DataFrame(zip(y_true3, y_true4)))
        pred = scale.inverse_transform(pd.DataFrame(zip(y_pred3, y_pred4)))

        lat_rmse = mean_squared_error(true[:,0], pred[:,0], squared=False) 
        longitude_rmse = mean_squared_error(true[:,1], pred[:,1], squared=False)

    print('Accuracy of the model for Floor : {:.2f} %'.format(floor_acc*100))
    print('Accuracy of the model for Building : {:.2f} %'.format(building_acc*100))
    print('RMSE of the model for latitude : {:.2f}'.format(lat_rmse))
    print('RMSE of the model for longitude : {:.2f}'.format(longitude_rmse))
