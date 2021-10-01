# -*- coding: utf-8 -*-

# Custom Imports 
from init import *
from dataset import *
from training import *
from evaluation import *
from model import *
from scaling import *

# Load and Process Data
print('Processing Training & Validation Set...')
training = pd.read_csv('trainingData.csv')
validation = pd.read_csv('validationData.csv')

training = training.iloc[:,:524]
validation = validation.iloc[:,:524]

target = ['FLOOR','BUILDINGID','LATITUDE','LONGITUDE']
to_be_scaled = ['LATITUDE','LONGITUDE']
num_floor = training['FLOOR'].nunique()
num_building = training['BUILDINGID'].nunique()

print('Scaling targets...')
training, scale = scaling(training, to_be_scaled, mode='train')
validation, _ = scaling(validation, to_be_scaled, mode='valid', scale=scale)

# Preparing the datasets for training
print('Preparing datasets for training...')
X_train = training.drop(target, axis=1)
y_train = training[target]

X_val = validation.drop(target, axis=1)
y_val = validation[target]

train = dataset(torch.from_numpy(np.array(X_train)).float(), torch.from_numpy(np.array(y_train)).float())
val = dataset(torch.from_numpy(np.array(X_val)).float(), torch.from_numpy(np.array(y_val)).float())

# Create the model: 
model = Model(n_feature= X_train.shape[1], num_floor = num_floor, num_building = num_building)

# Choose the hyperparameters for training: 
num_epochs = 25
batch_size = 10
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

#Loss function
criterion2 = nn.MSELoss()
criterion1 = nn.CrossEntropyLoss()

# Checking if gpu is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training the model
print('Training the model...')
best_model = fit(num_epochs, batch_size, criterion1, criterion2, optimizer, model, train, val, device=device, scheduler=scheduler)

print('Results...')
accuracy(val, best_model, scale=scale)

