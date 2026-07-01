import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from google.colab import drive


from google.colab import drive
drive.mount('/content/drive')


train = pd.read_csv('/content/drive/KDDTrain+.txt', header=None)
test = pd.read_csv('/content/drive/KDDTest+.txt', header=None)

# Data info
print("Train Data Info:")
print(train.info())
print("Test Data Info:")
print(test.info())

# First 5 rows
print("First 5 rows of Train Data:")
print(train.head())

# Last 5 rows
print("Last 5 rows of Train Data:")
print(train.tail())
