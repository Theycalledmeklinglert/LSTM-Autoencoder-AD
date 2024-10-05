import numpy as np
import tensorflow as tf
import pandas as pd
import warnings

from data_processing import clean_csv

warnings.filterwarnings('ignore')
import seaborn as sns

from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 16, 6 # set figsize for all images

np.random.seed(1)
tf.random.set_seed(1)

print('Tensorflow version:', tf.__version__)


"https://github.com/datablogger-ml/Anomaly-detection-with-Keras/blob/master/Anomaly_Detection_Time_Series.ipynb"

if __name__ == '__main__':
    df = clean_csv("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-current_steering_angle.csv")
    df.drop(columns=['Anomaly'], inplace=True)
    df.info()


    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df['Time'], y=df['data'], mode='lines', name='close'))  # lines mode for lineplot
    # fig.update_layout(title='16_05_23 can_interface-current_steering_angle', xaxis_title="Time", yaxis_title='data', showlegend=True)
    # fig.show()


