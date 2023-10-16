import os

import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans

from Meme import Meme


def main():

    memes: list[Meme] = initialize_memes()  # All meme curves and their data
    # For the 4 trend types, will be used as the number of clusters
    #trend_types = os.listdir('Dataset')
    #purge_ds_store(trend_types)

    for meme in memes:
        meme.normalize_ranks()
        meme.preprocess_ranks()
        #meme.save_curve_plot(False, meme.name)
        print("meme: " + meme.name + " (" + str(len(meme.ranks)) + ")")
        #print("ranks: \n" + str(meme.ranks))

    all_meme_ranks = [meme.ranks for meme in memes]
    # all_meme_labels = [meme.trend_type for meme in memes]
    # all_meme_label_indexes = str([trend_types.index(trend) for trend in all_meme_labels])

    kmeans = TimeSeriesKMeans(n_clusters=4, metric="softdtw", max_iter=400, random_state=0)
    kmeans.fit(all_meme_ranks)

    for i, label in enumerate(kmeans.labels_):
        memes[i].cluster = label

    print(f'Labels:\n{str(kmeans.labels_)}')
    for i in range(4):
        print(f'Cluster {i} Entries: {len([1 for n in kmeans.labels_ if n == i])}')

    memes.sort(key=lambda a_meme: a_meme.cluster) # Sorts memes by cluster #; ascending order
    plot_all_memes_of_each_type(memes)


def initialize_memes() -> list[Meme]:
    meme_names = os.listdir('Dataset')
    purge_ds_store(meme_names)

    memes: list[Meme] = []

    for meme_name in meme_names:
        meme_path = os.path.join('Dataset', meme_name)
        new_meme = Meme(meme_path, "blank")
        memes.append(new_meme)

    return memes


# def initialize_memes_old() -> list[Meme]:
#     # File paths to all meme type folders
#     purge_ds_store(Meme.meme_types)
#     memes: list[Meme] = []
#
#     for meme_type in Meme.meme_types:
#         meme_type_path = os.path.join('Dataset', meme_type)
#
#         meme_csvs = os.listdir(meme_type_path)
#         purge_ds_store(meme_csvs)
#         for meme_csv in meme_csvs:
#             meme_csv_path = os.path.join(meme_type_path, meme_csv)
#             new_meme = Meme(meme_csv_path, meme_type)
#             memes.append(new_meme)
#
#     return memes


def purge_ds_store(paths: list[str]) -> None:
    for path in paths:
        if path == '.DS_Store':
            paths.remove(path)


def plot_all_memes_of_each_type(memes: list[Meme]):
    plt.clf()
    trend_type = memes[0].cluster
    plt.title(f'Ranking of Cluster "{trend_type}" Memes Over time')
    plt.ylabel("Meme Ranking")
    plt.xlabel("Months")

    last_meme_index = len(memes) - 1
    index = 0
    for meme in memes:
        if meme.cluster != trend_type or index == last_meme_index:
            plt.savefig(os.path.join('Plots', 'dtwkmeans', f'all_{trend_type}_memes.png'))
            trend_type = meme.cluster
            plt.clf()
            plt.title(f'Ranking of "Cluster {trend_type}" Memes Over time')
            plt.ylabel("Meme Ranking")
            plt.xlabel("Months")

        plt.plot(Meme.months, meme.ranks)
        index += 1


if __name__ == '__main__':
    main()
