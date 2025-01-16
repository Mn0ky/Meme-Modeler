import os

import pandas as pd

from Countries import Countries
from DataUtils import DataUtils


class RegionDataMeme:
    countries: list[str] = Countries.countries
    countries_len = len(countries)
    countries_range: range = range(countries_len)

    def __init__(self, csv_path: str):
        self.csv_path: str = csv_path
        self.meme_name: str = os.path.basename(csv_path).replace(".csv", "")
        self.weighted_links: list[list] = self.read_meme_csv()

    # Returns an edge list CSV from a computed adjacency matrix for a meme's region data
    def read_meme_csv(self) -> list[list]:
        csv_data = pd.read_csv(self.csv_path, sep='\t')
        weighted_links = self.compute_weighted_links(csv_data)

        print(f"Meme: {self.meme_name}")
        print(weighted_links)
        return weighted_links

    def compute_weighted_links(self, csv_data) -> list[list]:
        weighted_links = []

        for country_x in self.countries_range:
            country_x_interest = float(csv_data.iloc[country_x, 1])
            country_x_name = Countries.countries[country_x]
            country_x_data = DataUtils.get_saved_data(country_x_name)

            # Calculate weight for country x
            weight: float = (country_x_interest
                             * DataUtils.get_ref_meme_interest(country_x)
                             * country_x_data['user_count']  # Number of internet users\
                             * country_x_data['market_share'])  # Google market share

            if weight > 0:  # Only want nonzero weights
                weighted_links.append([country_x_name, self.meme_name, weight])

        return weighted_links

    # Normalizing the rank values between 0 and 1 by utilizing SciKits MinScaler
    def normalize_ranks(self) -> None:
        pass

    def preprocess_ranks(self) -> None:
        self.normalize_ranks()

