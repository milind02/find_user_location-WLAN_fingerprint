# -*- coding: utf-8 -*-

# Basic Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


# Pytorch Utilities
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


