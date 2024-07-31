# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
from tensorflow import keras

from tf_lstm_autoencoder import test_lstm_autoencoder


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':

    #test_stacked_LSTM("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-current_steering_angle.csv")
    #old_LSTM_autoencoder("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv")
    #grid_search_LSTM_autoencoder("./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv")


    print(tf.__version__)
    print(keras.__version__)

    #test_lstm_autoencoder(10, [30, 30], 2, 0.0, 32, 120, ["./aufnahmen/csv/autocross_valid_16_05_23/can_interface-wheelspeed.csv", "./aufnahmen/csv/autocross_valid2_17_23_44/can_interface-wheelspeed.csv"])
    #test_lstm_autoencoder(10, [30, 30], 0.0, 32, 120, ["./aufnahmen/csv/autocross_valid_16_05_23", "./aufnahmen/csv/autocross_valid2_17_23_44"], './models/pretty good autoencoder for wheel speed/LSTM_autoencoder_decoder_30_30.keras')

    #get_sample_time("./aufnahmen/csv/autocross_valid_16_05_23")

    #test_lstm_autoencoder(10, [50, 50], 0.0, 32, 200, ["./aufnahmen/csv/csv test 1", "./aufnahmen/csv/csv test 2"]) #"C:\\Users\\Luca\\PycharmProjects\\AnoamlydetectionInFormulaStudent\\models\\LSTM_autoencoder_decoder_can_interface-wheelspeed_100_100.keras"

    #Test:
    #tune_lstm_autoencoder(30, ["./aufnahmen/csv/csv test 1", "./aufnahmen/csv/csv test 2"]) #todo: do time_steps = 30

    test = [[-1.13731737e+01, -7.83309102e-01, -3.89270544e-01, -1.23599720e+00, -1.44621856e+00],
 [-1.13563023e+01, -7.81260967e-01, -3.87439728e-01, -1.23333788e+00, -1.44439490e+00],
 [-1.13390446e+01, -7.79171228e-01, -3.85571599e-01, -1.23062372e+00, -1.44253356e+00],
 [-1.13213825e+01, -7.77038336e-01, -3.83665085e-01, -1.22785306e+00, -1.44063385e+00],
 [-1.13033123e+01, -7.74861455e-01, -3.81718874e-01, -1.22502542e+00, -1.43869538e+00],
 [-1.12848234e+01, -7.72639751e-01, -3.79732370e-01, -1.22213984e+00, -1.43671699e+00],
 [-1.12659130e+01, -7.70372272e-01, -3.77705336e-01, -1.21919441e+00, -1.43469758e+00],
 [-1.12465658e+01, -7.68057823e-01, -3.75635862e-01, -1.21618819e+00, -1.43263657e+00],
 [-1.12267675e+01, -7.65695453e-01, -3.73523831e-01, -1.21311951e+00, -1.43053277e+00],
 [-1.12065220e+01, -7.63284326e-01, -3.71367812e-01, -1.20998669e+00, -1.42838533e+00],
 [-1.11858101e+01, -7.60822415e-01, -3.69166732e-01, -1.20678902e+00, -1.42619379e+00],
 [-1.11646280e+01, -7.58309603e-01, -3.66920233e-01, -1.20352507e+00, -1.42395635e+00],
 [-1.11429605e+01, -7.55744815e-01, -3.64626527e-01, -1.20019341e+00, -1.42167242e+00],
 [-1.11207972e+01, -7.53126264e-01, -3.62285137e-01, -1.19679213e+00, -1.41934092e+00],
 [-1.10981312e+01, -7.50453353e-01, -3.59894872e-01, -1.19331932e+00, -1.41696067e+00],
 [-1.10749435e+01, -7.47724533e-01, -3.57454538e-01, -1.18977404e+00, -1.41453130e+00],
 [-1.10512362e+01, -7.44938135e-01, -3.54962826e-01, -1.18615413e+00, -1.41205056e+00],
 [-1.10269861e+01, -7.42093444e-01, -3.52419376e-01, -1.18245912e+00, -1.40951819e+00],
 [-1.10021887e+01, -7.39189506e-01, -3.49821806e-01, -1.17868578e+00, -1.40693231e+00],
 [-1.09768353e+01, -7.36223578e-01, -3.47169995e-01, -1.17483318e+00, -1.40429230e+00],
 [-1.09509039e+01, -7.33195186e-01, -3.44461918e-01, -1.17089880e+00, -1.40159698e+00],
 [-1.09243860e+01, -7.30103493e-01, -3.41696858e-01, -1.16688192e+00, -1.39884491e+00],
 [-1.08972654e+01, -7.26945877e-01, -3.38873148e-01, -1.16277933e+00, -1.39603455e+00],
 [-1.08695374e+01, -7.23720908e-01, -3.35989714e-01, -1.15859032e+00, -1.39316494e+00],
 [-1.08411818e+01, -7.20428228e-01, -3.33045244e-01, -1.15431166e+00, -1.39023466e+00],
 [-1.08121891e+01, -7.17065215e-01, -3.30037951e-01, -1.14994216e+00, -1.38724203e+00],
 [-1.07825356e+01, -7.13630557e-01, -3.26966882e-01, -1.14547908e+00, -1.38418622e+00],
 [-1.07522202e+01, -7.10122228e-01, -3.23830247e-01, -1.14092112e+00, -1.38106520e+00],
 [-1.07212229e+01, -7.06538558e-01, -3.20626616e-01, -1.13626516e+00, -1.37787743e+00],
 [-1.06895256e+01, -7.02878237e-01, -3.17354321e-01, -1.13150930e+00, -1.37462194e+00],
 [-1.06571121e+01, -6.99138999e-01, -3.14012170e-01, -1.12665141e+00, -1.37129719e+00],
 [-1.06239700e+01, -6.95319533e-01, -3.10598016e-01, -1.12168801e+00, -1.36790092e+00],
 [-1.05900803e+01, -6.91417456e-01, -3.07110906e-01, -1.11661768e+00, -1.36443205e+00],
 [-1.05554276e+01, -6.87430859e-01, -3.03548217e-01, -1.11143756e+00, -1.36088808e+00],
 [-1.05199976e+01, -6.83357239e-01, -2.99908996e-01, -1.10614479e+00, -1.35726852e+00],
 [-1.04837646e+01, -6.79195642e-01, -2.96190977e-01, -1.10073662e+00, -1.35357065e+00],
 [-1.04467192e+01, -6.74942732e-01, -2.92392492e-01, -1.09521079e+00, -1.34979279e+00],
 [-1.04088345e+01, -6.70596719e-01, -2.88511515e-01, -1.08956373e+00, -1.34593351e+00],
 [-1.03700914e+01, -6.66155338e-01, -2.84546494e-01, -1.08379292e+00, -1.34199030e+00],
 [-1.03304768e+01, -6.61616445e-01, -2.80494809e-01, -1.07789481e+00, -1.33796210e+00],
 [-1.02899656e+01, -6.56977654e-01, -2.76355028e-01, -1]]


    test_lstm_autoencoder(100, [120], 0.0, 32, 500, ["./aufnahmen/csv/csv test 1", "./aufnahmen/csv/csv test 2"], "./models/LSTM_autoencoder_decoder_can_interface-wheelspeed_timesteps100_layers_120.keras")
    #df = clean_csv("C:\\Users\\Luca\\PycharmProjects\AnoamlydetectionInFormulaStudent\\aufnahmen\csv\\autocross_valid_16_05_23\\diagnostics.csv")
    #print_unique_values(df, "status")


    print_hi('PyCharm')

    #todo: Note if I want to change the amount of timesteps, a new model has to be trained on it

