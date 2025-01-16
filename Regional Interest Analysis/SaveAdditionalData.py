import os
import pandas as pd
import json

from Countries import Countries
from DataUtils import DataUtils


# Saves country internet user count and google market share for reuse
def main():
    # Sourced from World Bank Data and statcounter Global Stats
    market_and_user_data = {
        'countries': []
    }

    for country in Countries.countries:
        user_count = DataUtils.get_country_user_count(country)
        market_share = DataUtils.get_country_google_mkt_share(country)

        country_data = {
            'name': country,
            'user_count': user_count,
            'market_share': market_share
        }

        print(country_data)

        market_and_user_data['countries'].append(country_data)

    with open('market_and_user_data.json', 'w') as fi:
        json.dump(market_and_user_data, fi, indent=4)


if __name__ == '__main__':
    main()
