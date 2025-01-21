import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import linear_regression.functions as f

data_frame = pd.read_csv(r"datasets\\df_regresion_lineal_1.csv")

output = data_frame['y'].to_numpy()

# delete Y from data_frame
data_frame.drop(columns=['y'], inplace=True)

X = data_frame.to_numpy()

w = f.pseudoinverse_method(X,output)

f.plot_linear_regression(X, output, w)
plt.show()