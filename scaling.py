# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
import pandas as pd

def scaling(dataset, features, mode='train', scale=None):

    features1 = dataset[features]

    if mode=='train':
        scale = StandardScaler()
        scale.fit(features1)

    features1 = scale.transform(features1)

    dataset = pd.concat([dataset.drop(features, axis=1), pd.DataFrame(features1, columns=features).astype(float)], axis=1)

    return dataset, scale
