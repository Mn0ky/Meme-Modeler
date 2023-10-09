import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from Meme import Meme


def main():
    # Want months represent numerically, in order to fit on y-axis
    memes: list[Meme] = initialize_memes()

    for meme in memes:
        print("reading rankings for meme: " + meme.name + " of type: " + meme.trend_type)
        meme.normalize_ranks()
        #meme.save_curve_plot(False, 'Normalized', f'{meme.name}_normalized.png')

    plot_all_memes_of_each_type(memes)

    print("finished!!")


def initialize_memes() -> list[Meme]:
    # File paths to all meme type folders
    purge_ds_store(Meme.meme_types)
    memes: list[Meme] = []

    for meme_type in Meme.meme_types:
        meme_type_path = os.path.join('Dataset', meme_type)

        meme_csvs = os.listdir(meme_type_path)
        purge_ds_store(meme_csvs)
        for meme_csv in meme_csvs:
            meme_csv_path = os.path.join(meme_type_path, meme_csv)
            new_meme = Meme(meme_csv_path, meme_type)
            memes.append(new_meme)

    return memes


def purge_ds_store(paths: list[str]) -> None:
    for path in paths:
        if path == '.DS_Store':
            paths.remove(path)


def plot_all_memes_of_each_type(memes: list[Meme]):
    plt.clf()
    trend_type = memes[0].trend_type
    plt.title(f'Ranking of "{trend_type}" Memes Over time')
    plt.ylabel("Meme Ranking")
    plt.xlabel("Months")

    for meme in memes:
        if meme.trend_type != trend_type:
            plt.savefig(os.path.join('Plots', 'Normalized', f'all_{trend_type}_memes_normalized.png'))
            trend_type = meme.trend_type
            plt.clf()
            plt.title(f'Ranking of "{trend_type}" Memes Over time')
            plt.ylabel("Meme Ranking")
            plt.xlabel("Months")

        plt.plot(Meme.months, meme.ranks)


if __name__ == '__main__':
    main()
