# This is a sample Python script.
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from IPython.display import display  #
from scipy.stats import norm
import statistics
from scipy.interpolate import *


def run():
    x_data, y_data = generate_a_dataset()
    df = generate_a_dataframe(x_data, y_data)
    df_sorted = sort_dataframe(df)
    display(df_sorted)
    plot_density_of_domain(df) # Plottet Dichte mit Werte von X um eine verteilung Daten auszugeben
    x, y = get_data_from_df(df_sorted)
    plot_data(x, y)

    # einfache interpolation
    'GENAU VERSTEHEN WIE DIE REGERESSION HIER FUNKTIONIERT ---> KLEINSTER QUAD FEHLER'
    # f1 = np.polyfit(x, y, 1)
    f2 = np.polyfit(x, y, 2)
    f3 = np.polyfit(x, y, 3)

    fig, ax = plt.subplots(1, 1)
    # ax.plot(x, np.polyval(f2,x), '-', x, np.polyval(f3,x), '--')
    ax.scatter(x_data, y_data, s=2)
    ax.plot(x, (x - 5) ** 2, color="black")

    ax.plot(x, np.polyval(f2, x), '-.', x, np.polyval(f3, x), '--')
    plt.show()


def generate_a_dataset():
    x_data = np.random.rand(1000) * 10
    y_data = function(x_data)
    return x_data, y_data


def function(x):
    # NORMALVERTEILTER GAUS
    noise = np.random.normal(0, 1, len(x))
    # GLEICHVERTEILTE STÖRUNG
    #noise= np.random.rand(len(x))
    y_data = (x - 5) ** 2 + noise  # function

    return y_data


def plot_data(x_data, y_data):
    plt.figure()
    plt.scatter(x_data, y_data, s=2)
    plt.show()

def difference_plot(x_data, y_data):
    plt.figure(3)
    plt.scatter(x_data, y_data, s=2)
    plt.show()

def generate_a_dataframe(x_data, y_data):
    d = {'X': x_data, 'Y': y_data}
    df = pd.DataFrame(data=d)
    return df


def sort_dataframe(df):
    final_df = df.sort_values(by=['X'], ascending=True)
    return final_df


def get_data_from_df(df):
    x = df['X']
    y = df['Y']
    return x, y

def plot_density_of_domain(df):
    values_of_domain = get_values_of_domain(df)
    values_of_domain.plot(kind="hist")
    values_of_domain.plot.kde()

#TODO: "BEKOMME ICH HIER DIE FUNKTION RAUS ?"
def get_values_of_domain(df):
    column_of_df = df.columns.values.tolist()
    del column_of_df[-1]  # column_of_df[:-1]
    values_of_domain = df[column_of_df]
    return values_of_domain


run()