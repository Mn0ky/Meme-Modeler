import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


class Meme:
    months: list[int] = list(range(120))  # All meme datasets span the course of 120 months
    meme_types: list[str] = os.listdir('Dataset')
    _scaler = MinMaxScaler(feature_range=(0, 100), copy=False)

    def __init__(self, csv_path: str, trend_type: str):
        self.csv_path: str = csv_path
        self.ranks: np.array[int] = self.read_meme_csv()  # Y-axis Values
        self.trend_type: str = trend_type
        self.name: str = os.path.basename(csv_path).replace(".csv", "")
        self.cluster = -1

    # Returns the month-by-month ranks of a meme
    def read_meme_csv(self) -> np.ndarray[int]:
        # Column 2 is meme rankings. Skip first 2 rows to get to start of column
        rank_col = pd.read_csv(self.csv_path, usecols=[1], skiprows=1)
        # Clean the ranking data to have only numeric values, then return them
        return rank_col.replace('<1', 0).iloc[:, 0].astype(int).values

    # Normalizing the rank values between 0 and 1 by utilizing SciKits MinScaler
    def normalize_ranks(self) -> None:
        self.ranks = self.ranks.reshape(len(self.ranks), 1)  # Must be 2D
        self.ranks = Meme._scaler.fit_transform(self.ranks)  # Fit and apply normalization
        self.ranks = self.ranks.flatten()  # Back to 1D
        # print(f'{self.name} max is {Meme.scaler.data_max_}')
        # print(f'{self.name} min is {Meme.scaler.data_min_}')
        # Those with min's of 0 and max's of 100 will simpy be the same values but divided by 100 (i.e 50 -> .5)
        # print(f'{self.name} normalized values are:\n{str(self.ranks)}')

    #
    def smooth_ranks(self) -> None:
        # self.ranks = self.ranks.reshape(len(self.ranks), 1)  # Must be 2D
        # self.ranks = TimeSeriesScalerMeanVariance().fit_transform(self.ranks)  # Apply mean scaling
        # self.ranks = self.ranks.flatten()  # Back to 1D
        print(f'{self.name} has {len(self.ranks)} ranks:\n{str(self.ranks)}')
        # Window of 6, don't want to smooth too much
        self.ranks = pd.Series(self.ranks).rolling(6).mean().fillna(0).to_numpy()
        #print(str(self.ranks))

    # Attempts to "capture the period of the trend"
    def preprocess_ranks(self) -> None:
        self.smooth_ranks()
        self.normalize_ranks()
        self.capture_trend_curve()

    def capture_trend_curve(self) -> None:
        total_period: int = 43

        if len(Meme.months) != total_period:  # Number of months should equal total period
            Meme.months = list(range(total_period))

        peak_index = np.argmax(self.ranks)  # Index of peak (max)

        offset = 1
        popularity_start_rank = 25
        previous_rank = self.ranks[peak_index]  # Should be 100 (peak), if normalization was applied

        np.argmax(self.ranks < 25)

        while previous_rank > popularity_start_rank:
            # No rank of 25 or less was found or trend portion would not include peak
            peak_difference = peak_index - offset

            if peak_difference < 0 or peak_difference + total_period < peak_index:
                popularity_start_rank += 1
                offset = 1
                continue

            previous_rank = self.ranks[peak_index - offset]
            offset += 1

        trend_start_index = peak_index - offset
        trend_end_index = trend_start_index + total_period
        if trend_end_index > len(self.ranks) - 1:
            print("meme too short: " + self.name)
            # Account for possibility of trend curve being less than 43 months
            trend_start_index -= abs(len(self.ranks) - trend_end_index)

        self.ranks = self.ranks[trend_start_index:trend_end_index]
        return

    # Saves a graph of the meme trend curve to the Plots folder
    def save_curve_plot(self, show_plot: bool, *argv: str) -> None:
        plt.clf()
        plt.title(f'Ranking of "{self.name}" Meme Over time')
        plt.ylabel("Meme Ranking")
        plt.xlabel("Months")
        plt.plot(Meme.months, self.ranks)
        # plot_path = os.path.join('Plots', self.t, meme_csv)
        plt.savefig(os.path.join('Plots', *argv))

        if show_plot:
            plt.show()
