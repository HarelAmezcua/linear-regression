import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import linear_regression.functions as f

# Imports the dataframe
data_frame = pd.read_csv(r"datasets\\df_regresion_lineal_1.csv")

X = data_frame['x'].to_numpy()
Y = data_frame['y'].to_numpy()

w = f.least_squares(X, Y)

f.plot_linear_regression(X, Y, w)
plt.show()