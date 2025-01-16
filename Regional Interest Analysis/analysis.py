import os

import pandas as pd
#from sklearn.preprocessing import MinMaxScaler

from RegionDataMeme import RegionDataMeme


def main():
    print('Initializing memes...')
    memes = initialize_memes()

    # Combine weighted links
    print('Combining weighted links...')
    all_weighted_links = []
    for meme in memes:
        all_weighted_links.extend(meme.weighted_links)

    print('Normalizing weights...')
    normalize_weights(all_weighted_links)

    print('Saving weighted link list...')
    weighted_link_csv = pd.DataFrame(all_weighted_links, columns=['Source', 'Target', 'Weight'])
    weighted_link_csv.index.name = 'ID'

    save_path = '../Dataset/regional_interest/'
    save_path = os.path.join(save_path, 'weighted_links.csv')
    #pd.set_option('display.float_format', lambda x: f'{x:.20f}')
    weighted_link_csv.to_csv(save_path, index=True, float_format='%.8f')


def normalize_weights(weighted_links: list[list]) -> None:
    all_weights = [weighted_link[2] for weighted_link in weighted_links]
    min_weight = min(all_weights)
    max_weight = max(all_weights)

    for weighted_link in weighted_links:
        weight = weighted_link[2]

        standardized_weight = scale_weight(weight, min_weight, max_weight)
        weighted_link[2] = standardized_weight


def scale_weight(weight, min_weight, max_weight) -> float:
    return (weight - min_weight) / (max_weight - min_weight)


def initialize_memes() -> list[RegionDataMeme]:
    """
    Meme's listed that had google market share set at default due to lack of data:
    british-indian-ocean-territory
    caribbean-netherlands
    cocos-keeling-islands
    curacao
    french-southern-territories
    heard-mcdonald-islands
    kosovo
    sint-maarten
    south-georgia-south-sandwich-islands
    south-sudan
    saint-barthelemy
    saint-martin
    svalbard-and-jan-mayen
    us-outlying-islands

    Meme's listed that had percentage of internet users share set at default due to lack of data:
    American samoa
    """

    # TODO: Finish region dataset #3 and move region data to Dataset dir
    meme_names = os.listdir(
        '../KnowYourMemeCollector/gtrends_data/regional_interest/RI Dataset 2 - [Earliest Origin, Low Vol Countries]/')
    purge_ds_store(meme_names)

    memes: list[RegionDataMeme] = []

    for meme_name in meme_names:
        meme_path = os.path.join('../KnowYourMemeCollector/gtrends_data/regional_interest/RI Dataset 2 - [Earliest '
                                 'Origin, Low Vol Countries]',
                                 meme_name)

        if not os.path.isfile(meme_path):
            continue

        new_meme = RegionDataMeme(meme_path)
        memes.append(new_meme)

    return memes


def purge_ds_store(paths: list[str]) -> None:
    for path in paths:
        if path == '.DS_Store':
            paths.remove(path)


if __name__ == '__main__':
    main()
