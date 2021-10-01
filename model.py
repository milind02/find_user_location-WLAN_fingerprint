# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    """
    Model based on a four layer neural architecture for each output
    
    Parameters
    ----------
    n_feature: number of input features
    num_floor: number of floor categories
    num_building: number of building categories

    Returns
    -------
    floor, building, latitude, longitude
    """
    
    def __init__(self, n_feature, num_floor, num_building):
        super(Model, self).__init__()

        #Only first layer is shared
        self.linear = nn.Linear(n_feature, 128)

        self.linear2_floor = nn.Linear(128, 64)
        self.linear3_floor = nn.Linear(64, 32)

        self.linear2_building = nn.Linear(128, 64)
        self.linear3_building = nn.Linear(64, 32)

        self.linear2_long = nn.Linear(128, 64)
        self.linear3_long = nn.Linear(64, 32)

        self.linear2_lat = nn.Linear(128, 64)
        self.linear3_lat = nn.Linear(64, 32)
        
        #Output layers
        self.l_floor = nn.Linear(32, num_floor)
        self.l_building = nn.Linear(32, num_building)
        self.l_longitude = nn.Linear(32, 1)
        self.l_latitude = nn.Linear(32, 1)
        
    def forward(self, inputs):

        p = F.relu(self.linear3_floor(F.relu(self.linear2_floor(F.relu(self.linear(inputs))))))
        q = F.relu(self.linear3_building(F.relu(self.linear2_building(F.relu(self.linear(inputs))))))
        r = F.relu(self.linear3_long(F.relu(self.linear2_long(F.relu(self.linear(inputs))))))
        s = F.relu(self.linear3_lat(F.relu(self.linear2_lat(F.relu(self.linear(inputs))))))

        floor = F.softmax(self.l_floor(p), dim=1)
        building = F.softmax(self.l_building(q), dim=1)
        latitude = self.l_latitude(r)
        longitude = self.l_longitude(s)

        return (floor, building, latitude, longitude)
