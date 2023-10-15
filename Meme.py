import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


class Meme:
    months: list[int] = list(range(120))  # All meme datasets span the course of 120 months
    meme_types: list[str] = os.listdir('Dataset')
    _scaler = MinMaxScaler(feature_range=(0, 100), copy=False)

    def __init__(self, csv_path: str, trend_type: str):
        self.csv_path: str = csv_path
        self.ranks: np.ndarray[int] = self.read_meme_csv()  # Y-axis Values
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

    # While not a Bell curve, eventually try standardization to compare with normalization
    def standardize_ranks(self) -> None:
        pass

    # Attempts to "capture the period of the trend"
    def preprocess_ranks(self) -> None:
        total_period: int = 43

        if len(Meme.months) != total_period:  # Number of months should equal total period
            Meme.months = list(range(total_period))

        for i, rank in enumerate(self.ranks):
            if rank == 100:
                peak_index = i

                offset = 1
                popularity_start_rank = 25
                previous_rank = rank  # Will be 100, since that is the peak
                while previous_rank > popularity_start_rank:
                    # No rank of 25 or less was found or trend portion would not include peak
                    peak_difference = peak_index - offset
                    if peak_difference < 0 or peak_difference + total_period < peak_index:
                        popularity_start_rank += 1
                        offset = 1

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

        # index: int = 0
        # offset_start: int = -1
        # total_period_end: int = -1
        # for rank in self.ranks:
        #     if rank >= threshold_rank:
        #         offset_start = max(index - offset, 0)  # Assigns starting index of offset; 0 if negative
        #         # Accounts for time before threshold rank; just the remaining indices if total period > remaining indices
        #         total_period_end = min(total_period + offset_start, len(self.ranks) - 1)
        #
        #         if total_period_end == len(self.ranks) - 1:
        #             offset_start -= total_period + offset_start - total_period_end  # Keep number of months the same
        #         break
        #
        #     index += 1
        #

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
