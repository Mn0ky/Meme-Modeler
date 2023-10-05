import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reads in data from a CSV
aliens_data = pd.read_csv('Dataset\Ancient Aliens.csv', usecols = ['Month', 'Ancient Aliens: (United States)'],
                          skiprows=2)
#aliens_data.head() # TODO: Change this as it only reads first 5 rows of data!

monthes = aliens_data.loc[:, 'Month'] # X-axis values, use iloc later instead of string value
monthes = [month for month in range(len(monthes))] # Want monthes represent numerically, in order to fit on y-axis
num_memes = aliens_data.loc[:, 'Ancient Aliens: (United States)'] # Y-axis values, use  iloc later instead of string value

plt.xlabel("Month")
plt.ylabel("Number of Memes")
plt.title("Number of Ancient Aliens Memes Over time") #
plt.scatter(monthes, num_memes)
plt.show()