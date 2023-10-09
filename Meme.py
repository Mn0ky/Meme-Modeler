import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Meme:
    months: list[int] = list(range(120))  # All meme datasets span the course of 120 months
    meme_types: list[str] = os.listdir('Dataset')

    def __init__(self, csv_path: str, trend_type: str):
        self.csv_path: str = csv_path
        self.ranks: pd.DataFrame = self.read_meme_csv()  # Y-axis Values
        self.trend_type: str = trend_type
        self.name: str = os.path.basename(csv_path).replace(".csv", "")

    # Returns the month-by-month ranks of a meme
    def read_meme_csv(self) -> np.ndarray[int]:
        # Column 2 is meme rankings. Skip first 2 rows to get to start of column
        rank_col = pd.read_csv(self.csv_path, usecols=[1], skiprows=2)
        # Clean the ranking data to have only numeric values, then return them
        return rank_col.replace('<1', 0).iloc[:, 0].astype(int).values

    # Attempts to normalize the curves via a threshold rank, offset, and total period
    def normalize_ranks(self) -> None:
        threshold_rank: int = 90
        offset: int = 10
        total_period: int = 60

        if len(Meme.months) != total_period:  # Number of months should equal total period
            Meme.months = list(range(total_period))

        index: int = 0
        offset_start: int = -1
        total_period_end: int = -1
        for rank in self.ranks:
            if rank >= threshold_rank:
                offset_start = max(index - offset, 0)  # Assigns starting index of offset; 0 if negative
                # Accounts for time before threshold rank; just the remaining indices if total period > remaining indices
                total_period_end = min(total_period + offset_start, len(self.ranks) - 1)

                if total_period_end == len(self.ranks) - 1:
                    offset_start -= total_period + offset_start - total_period_end  # Keep number of months the same
                break

            index += 1

        self.ranks = self.ranks[offset_start:total_period_end]

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
