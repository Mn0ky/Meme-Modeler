import os
import shutil

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from Meme import Meme


# from tslearn.barycenters import softdtw_barycenter


def main():
    # For the 4 trend types, will be used as the number of clusters
    # trend_types = os.listdir('Dataset')
    # purge_ds_store(trend_types)

    is_supervised = True if input("Supervised (0) or Unsupervised (1)") == '0' else False
    memes = initialize_labeled_memes() if is_supervised else initialize_memes()  # All meme curves and their data

    for meme in memes:
        meme.preprocess_ranks()
        # if len(meme.ranks) != 43:
        #     print(f'{meme.name} has {len(meme.ranks)} ranks:\n{str(meme.ranks)}')
        # meme.save_curve_plot(False, meme.name)
        print("meme: " + meme.name + " (" + str(len(meme.ranks)) + ")")
        # print("ranks: \n" + str(meme.ranks))

    all_meme_ranks = [meme.ranks for meme in memes]

    if is_supervised:
        memes.sort(key=lambda a_meme: a_meme.cluster)  # Sorts memes by cluster #; ascending order
        clf = svm.SVC(kernel='linear', probability=True)
        all_ranks = [meme.ranks for meme in memes]
        all_cluster_targets = [meme.cluster for meme in memes]

        training_ranks, testing_ranks, training_cluster_targets, testing_cluster_targets = \
            train_test_split(all_ranks, all_cluster_targets, test_size=0.2)

        clf.fit(training_ranks, training_cluster_targets)
        testing_results = clf.predict(testing_ranks)

        print('Generating classification report...')
        print(classification_report(testing_cluster_targets, testing_results))
        print(all_cluster_targets)
        y_pred = clf.predict_proba(testing_ranks)
        cross_entropy_loss = log_loss(testing_cluster_targets, y_pred)
        print(f'cross_entropy_loss: {cross_entropy_loss}')
        roc_auc_score = metrics.roc_auc_score(testing_cluster_targets, y_pred, multi_class='ovo')
        print(f'roc_auc_score: {roc_auc_score}')
        return
    # all_meme_labels = [meme.trend_type for meme in memes]
    # all_meme_label_indexes = str([trend_types.index(trend) for trend in all_meme_labels])

    # sample_cluster_counts = [3, 4, 5, 6, 7, 8, 9, 10]
    # optimal_c_count = get_optimal_cluster_count(sample_cluster_counts, all_meme_ranks)
    # print(f'Optimal Cluster Count is: {optimal_c_count} clusters.')

    kmeans = KMeans(init="k-means++", n_clusters=4, n_init=400, random_state=0)
    kmeans.fit(all_meme_ranks)

    for i, label in enumerate(kmeans.labels_):
        memes[i].cluster = label

    print(f'Labels:\n{str(kmeans.labels_)}')
    print(f'Total of {len(kmeans.labels_)} entries')
    for i in range(4):
        print(f'Cluster {i} Entries: {len([1 for n in kmeans.labels_ if n == i])}')

    memes.sort(key=lambda a_meme: a_meme.cluster)  # Sorts memes by cluster #; ascending order
    # create_scatter_plot(kmeans, memes)
    # plot_all_memes_of_each_type(memes)
    # create_labels(memes)
    print_category_distribution(memes)


# Performs silhouette analysis to compute "optimal" number of clusters. 4 for the current dataset.
def get_optimal_cluster_count(cluster_counts, training_data) -> int:
    highest_score = -1
    optimal_c_count = -1
    silhouette_scores: [int] = []

    for c_count in cluster_counts:
        kmeans = KMeans(init="k-means++", n_clusters=c_count, n_init=400, random_state=0)
        kmeans.fit(training_data)
        silhouette_avg = silhouette_score(training_data, kmeans.labels_)
        if silhouette_avg > highest_score:
            highest_score = silhouette_avg
            optimal_c_count = c_count
        silhouette_scores.append(silhouette_avg)
        print(f'Avg. Silhouette score for {c_count} clusters is: {silhouette_avg}')

    create_silhouette_graph(cluster_counts, silhouette_scores)
    return optimal_c_count


def create_silhouette_graph(cluster_counts, silhouette_scores) -> None:
    reset_plot("Silhouette Score vs. Number of Clusters", "Avg. Silhouette Score", "Number of Clusters")
    max_index = np.argmax(silhouette_scores)
    optimal_cluster = cluster_counts[max_index]
    optimal_silhouette_score = silhouette_scores[max_index]
    plt.vlines([optimal_cluster], 0, [optimal_silhouette_score], linestyle="dashed")
    plt.plot(cluster_counts, silhouette_scores, marker='o')
    # plt.show()
    plt.savefig(os.path.join('Plots', 'Misc.', 'silhouette_scores.png'))


def initialize_memes() -> list[Meme]:
    unlabeled_meme_data_path = os.path.join('Dataset', 'interest_over_time', 'Unlabeled Data')
    meme_names = os.listdir(unlabeled_meme_data_path)
    purge_ds_store(meme_names)

    memes: list[Meme] = []

    for meme_name in meme_names:
        meme_path = os.path.join(unlabeled_meme_data_path, meme_name)
        #print(meme_path)
        if not os.path.isfile(meme_path):
            continue
        new_meme = Meme(meme_path, -1)
        memes.append(new_meme)

    return memes


def initialize_labeled_memes() -> list[Meme]:
    # File paths to all meme type folders
    labeled_data_path = os.path.join('Dataset', 'interest_over_time', 'Labeled Data')
    label_paths = os.listdir(labeled_data_path)
    purge_ds_store(label_paths)
    memes: list[Meme] = []

    for meme_type in range(len(label_paths)):
        meme_type_path = os.path.join(labeled_data_path, f'Cluster {meme_type}')
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


def plot_all_memes_of_each_type(memes: list[Meme]) -> None:
    trend_type = memes[0].cluster
    reset_plot(f'Ranking of Cluster "{trend_type}" Memes Over time', 'Meme Ranking', 'Months')

    last_meme_index = len(memes) - 1
    current_clusters = []
    for i, meme in enumerate(memes):

        if meme.cluster != trend_type or i == last_meme_index:
            plt.plot(np.average(np.vstack(current_clusters), axis=0), c='red')  # Average curve of the trend cluster
            current_clusters = []
            plt.savefig(os.path.join('Plots', 'Total', f'all_{trend_type}_memes.svg'))
            trend_type = meme.cluster
            reset_plot(f'Ranking of Cluster "{trend_type}" Memes Over time', 'Meme Ranking', 'Months')

        plt.plot(Meme.months, meme.ranks, c='gray')
        current_clusters.append(meme.ranks)


def reset_plot(title: str, y_label: str, x_label: str) -> None:
    plt.clf()
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)


# def create_scatter_plot(kmeans_obj: KMeans, all_memes: list[Meme]) -> None:
#     # Getting the Centroids
#     reset_plot("Scatterplot of Resultant Clusters", '', '')
#     centroids = kmeans_obj.cluster_centers_
#     labels = kmeans_obj.labels_
#     u_labels = np.unique(labels)
#
#     # plotting the results:
#
#     # for i in u_labels:
#     #     plt.scatter([meme.ranks for meme in all_memes if meme.cluster == i], df[labels == i, 1], label=i)
#
#     plt.figure(figsize=(8, 6))
#         plt.scatter(Meme.months, [meme.cluster for meme in all_memes], c=kmeans_obj.labels_.astype(float))
#     plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
#     plt.legend()
#     plt.show()


def create_labels(memes: list[Meme]) -> None:
    labeled_data_path: str = os.path.join('Dataset', 'Labeled Data')
    if os.path.isdir(labeled_data_path):
        shutil.rmtree(labeled_data_path)

    os.mkdir(labeled_data_path)
    for i in range(4):
        label_folder_path = os.path.join(labeled_data_path, f'Cluster {i}')
        os.mkdir(label_folder_path)

    for meme in memes:
        meme_label_path: str = os.path.join(labeled_data_path, f'Cluster {meme.cluster}')
        shutil.copy(meme.csv_path, meme_label_path)


def print_category_distribution(all_memes: list[Meme]) -> None:
    cat_types = os.listdir('Categories')
    labeled_clusters_path = os.path.join('Dataset', 'interest_over_time', 'Labeled Data')
    labeled_clusters = os.listdir(labeled_clusters_path)
    purge_ds_store(cat_types)
    purge_ds_store(labeled_clusters)
    all_meme_names = [meme.name for meme in all_memes]

    # Calculate the number of rows and columns needed
    num_rows = len(cat_types) // 2 + len(cat_types) % 2
    num_rows -= 1
    num_cols = min(2, len(cat_types))

    # Use gridspec to handle subplot layout
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.8, hspace=0.15)

    plt.figure(figsize=(12, num_rows * 4))

    for i in range(len(labeled_clusters)):
        cat_label_sums: [int] = [0] * len(cat_types)  # 5 zeros
        for j in range(len(cat_types)):
            cur_cat_path = os.path.join('Categories', cat_types[j])
            cur_cat = os.listdir(cur_cat_path)
            purge_ds_store(cur_cat)

            for meme_name in cur_cat:
                meme_name = meme_name.replace('.csv', '')
                #print(f'looking for {meme_name}')
                if meme_name in all_meme_names:
                    #print(f'meme {meme_name} exists')
                    meme_index = all_meme_names.index(meme_name)
                    found_meme = all_memes[meme_index]
                    if found_meme.cluster == i:
                        cat_label_sums[j] += 1

        print(f'Category dist. for Cluster {i} ({sum(cat_label_sums)} Memes):')

        #cat_cluster_sizes: list[int] = []
        for j in range(len(cat_types)):
            num_cluster_memes = sum(cat_label_sums)
            #cat_cluster_sizes.append(num_cluster_memes)
            label_percentage = 100.0 * cat_label_sums[j] / num_cluster_memes
            print(f'\t"{cat_types[j]}" Category: {label_percentage:.1f}%')

        # Use subplots for each category
        row, col = divmod(i, num_cols)
        ax = plt.subplot(gs[row, col])

        ax.pie(cat_label_sums, labels=[f'"{cat}" Category' for cat in cat_types],
               autopct=lambda p: f'{p:.1f}%' if p > 0 else '')
        ax.set_title(f'{labeled_clusters[i].title()} Cluster Distribution')

    plt.savefig(os.path.join('Plots', 'Misc.', 'category_cluster_dist.svg'))


if __name__ == '__main__':
    main()
