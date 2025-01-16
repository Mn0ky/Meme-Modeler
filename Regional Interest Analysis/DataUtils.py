import json
import re
import unicodedata

import pandas as pd
import requests
from lxml import html

from Countries import Countries


class DataUtils:
    reference_meme_path: str = ('../Dataset/regional_interest/RI Dataset 2 - [Earliest Origin, Low Vol '
                                'Countries]/doge.csv')
    ref_meme_data = pd.read_csv(reference_meme_path, sep='\t')

    with open('market_and_user_data.json') as f:
        saved_data = json.load(f)

    @staticmethod
    def get_ref_meme_interest(country_idx) -> float:
        ref_meme_interest: float = DataUtils.ref_meme_data.iloc[country_idx, 1]

        return ref_meme_interest

    @staticmethod
    def get_saved_data(country_name) -> dict:
        country_data: list[dict] = DataUtils.saved_data['countries']

        for country in country_data:
            if country['name'] == country_name:
                return country

    @staticmethod
    def get_country_user_count(country_name):
        user_percentage: float = DataUtils._get_internet_users_percentage(country_name, 2014)
        population: int = DataUtils._get_country_population(country_name, 2014)
        user_percentage /= 100  # Convert to percentage decimal

        return user_percentage * population

    @staticmethod
    def get_country_google_mkt_share(country_name):
        country_name = DataUtils.slugify(country_name, allow_unicode=False)

        url = f'https://gs.statcounter.com/search-engine-market-share/all/{country_name}/#monthly-201401-201412-bar'
        print(f'{url}')

        # Send GET request to the webpage
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve data: {response.status_code}")

        # Parse the HTML content
        tree = html.fromstring(response.content)

        try:
            market_share = float(tree.xpath("//tbody/tr[th='Google']/td/span[@class='count']/text()")[0])
        except IndexError:
            market_share = 0.0

        return market_share

    @staticmethod
    def _get_internet_users_percentage(country, year) -> float:
        country_code = Countries.country_to_iso[country]
        url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/IT.NET.USER.ZS?date={year}&format=json"

        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Failed to retrieve data: {response.status_code}")

        data = response.json()

        if not data or len(data) < 2 or not data[1]:
            print(f'\n{response.text}\n')
            return 0

        # Extract the percentage of internet users
        percentage = data[1][0].get('value')

        if percentage is None:
            print(f'\n{response.text}')
            print("Data for the specified year is not available.\n")
            return 0

        return percentage

    @staticmethod
    def _get_country_population(country, year):
        country_code = Countries.country_to_iso[country]
        url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SP.POP.TOTL?date={year}&format=json"
        print(f'using url: {url}')

        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Failed to retrieve data: {response.status_code}")

        # Parse the JSON response
        data = response.json()

        # Extract the population data for the specified year
        if len(data) < 2 or not data[1]:
            print(f'\n{response.text}')
            print("No population data available for the given country and year.\n")
            return 0

        population_data = None
        try:
            population_data = data[1][0].get('value')
        except TypeError:
            print(f'\n{response.text}')
            print("No population data available for the given country and year.\n")
            return 0

        if population_data is None:
            print(f'\n{response.text}')
            print("No population data available for the given country and year.\n")
            return 0

        return population_data

    # noinspection DuplicatedCode
    @staticmethod
    def slugify(value, allow_unicode=True):
        """
            Taken from https://github.com/django/django/blob/master/django/utils/text.py

            Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
            dashes to single dashes. Remove characters that aren't alphanumerics,
            underscores, or hyphens. Convert to lowercase. Also strip leading and
            trailing whitespace, dashes, and underscores.

            Copyright (c) Django Software Foundation and individual contributors.
            All rights reserved.

            Redistribution and use in source and binary forms, with or without modification,
            are permitted provided that the following conditions are met:

                1. Redistributions of source code must retain the above copyright notice,
                   this list of conditions and the following disclaimer.

                2. Redistributions in binary form must reproduce the above copyright
                   notice, this list of conditions and the following disclaimer in the
                   documentation and/or other materials provided with the distribution.

                3. Neither the name of Django nor the names of its contributors may be used
                   to endorse or promote products derived from this software without
                   specific prior written permission.

            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
            ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
            WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
            DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
            ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
            (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
            LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
            ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
            (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
            SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
            """

        value = str(value)
        if allow_unicode:
            value = unicodedata.normalize('NFKC', value)
        else:
            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        value = re.sub(r'[^\w\s-]', '', value.lower())
        return re.sub(r'[-\s]+', '-', value).strip('-_')
