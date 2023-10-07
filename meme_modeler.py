import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    # File paths to all meme type folders
    meme_types: list[str] = os.listdir('Dataset')
    purge_ds_store(meme_types)

    # Want months represent numerically, in order to fit on y-axis
    months: list[int] = [month for month in range(120)]  # All memes span the course of 120 months

    for meme_type in meme_types:
        meme_type_path = os.path.join('Dataset', meme_type)

        meme_csvs = os.listdir(meme_type_path)
        purge_ds_store(meme_csvs)
        for meme_csv in meme_csvs:
            meme_csv_path = os.path.join(meme_type_path, meme_csv)
            plt.clf()
            plt.title(f'Ranking of "{meme_csv}" Meme Over time')
            plt.ylabel("Meme Ranking")
            plt.xlabel("Month")

            print("reading rankings for meme: " + meme_csv + " of type: " + meme_type)
            meme_ranks: list[int] = read_meme_csv(meme_csv_path)  # Y-axis values

            plt.plot(months, meme_ranks)
            plot_path = os.path.join('Plots', meme_type, meme_csv)
            plt.savefig(f'{plot_path}.png')


# Returns the month-by-month ranks of a meme
def read_meme_csv(csv_path: str) -> list[int]:
    # Column 2 is meme rankings. Skip first 2 rows to get to start of column
    rank_col = pd.read_csv(csv_path, usecols=[1], skiprows=2)
    # Clean the ranking data to have only numeric values, then return them
    return rank_col.replace('<1', 0).iloc[:, [0]].astype(int)


def purge_ds_store(paths: list[str]):
    for path in paths:
        if path == '.DS_Store':
            paths.remove(path)


if __name__ == '__main__':
    main()
