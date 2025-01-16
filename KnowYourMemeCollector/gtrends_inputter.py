import json
import os
import re
import unicodedata
import urllib.parse
from typing import Union

import pandas as pd
from pytrends.request import TrendReq
from requests.exceptions import RetryError


def main() -> None:
    pytrends = TrendReq(hl='en-US', tz=0, retries=3, backoff_factor=2)
    curwd = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(curwd, 'meme_keywords.json')

    with open(json_path) as f:
        json_data = json.load(f)

    cat_names: list[str] = json_data['categories']

    # Remove duplicates, decode any url encoded meme names, and get the data
    completed_memes: list[dict] = prior_completed_memes(curwd)
    gtrends_interest_type = input('What interest type would you like to request: Interest over time (0), Interest by '
                                  'Region (1)')
    gtrends_interest_type = int(gtrends_interest_type)

    for name in cat_names:
        cat_memes: list[dict] = json_data[name]

        for meme in cat_memes:
            meme_name = meme['name']

            if '%' in meme_name:
                decoded_meme_name = urllib.parse.unquote(meme_name)
                meme['name'] = decoded_meme_name
                print(f'decoded "{meme_name}" into "{decoded_meme_name}"')

            if check_if_completed(meme, completed_memes):
                print(f'Is completed meme "{meme_name}", skipping...')
                continue

            if is_error_meme(meme, cat_names, curwd):
                print(f'Is error meme "{meme_name}", skipping...')
                continue

            print('Getting gdata for meme ' + meme['name'])
            g_data = get_meme_data(meme, pytrends, gtrends_interest_type)

            if g_data is None:
                print(f'Unable to get data for {meme["name"]}, adding to errors list.')
                save_errored_memes(meme, name, curwd)
                continue

            save_meme_data(meme, name, g_data, gtrends_interest_type, curwd)
            completed_memes.append(meme)
            update_completed_memes_csv(completed_memes, curwd)


def get_meme_data(meme: dict, pytrends: TrendReq, interest_type: int) -> Union[pd.DataFrame, None]:
    kw_list = meme['keywords']  # ['Giorgio A. Tsoukalos', 'Giorgio Tsoukalos']
    year = meme['year']
    year_str = str(year - 2)  # Start 2 years before meme "got popular"
    year_stop = '2023'

    #year_str = format_year_range(year, year_stop, year_str)
    year_str = 'all'

    if kw_list is None:  # For memes without keywords
        return None
    if len(kw_list) == 1:
        pytrends.build_payload(kw_list, cat=0, timeframe=year_str)
        data = call_trends_api(pytrends, interest_type)
        print(f'Successfully got data for "{meme["name"]}"')
        return data

    # For memes with multiple keywords
    try:
        pytrends.build_payload(kw_list, cat=0, timeframe=year_str)
        data = call_trends_api(pytrends, interest_type)
        if data is None:
            return None

        max_kw = ''  # Can only use 1 keyword so that values are not in comparison with other keywords
        if interest_type == 0:
            # Gets column with max value between date and ispartial column
            max_kw = data.iloc[:, 1:-1].idxmax(axis=1).iloc[0]
        if interest_type == 1:
            # Gets column with the highest interest value; max_kw should be the same even if interest_type == 0
            max_kw = max(kw_list, key=lambda keyword: data[keyword].max())

        pytrends.build_payload([max_kw], cat=0, timeframe=year_str, gprop='')
        data = call_trends_api(pytrends, interest_type)  # GTrends data from "most popular" keyword
        return data
    except RetryError as ex:
        return None  # Sometimes build_payload fails due to it requesting token and then erroring out


def format_year_range(year, year_stop, year_str):
    if year <= 2004:
        year_str = 'all'
    elif year > 2021:
        year_str = str(2019)  # Make sure period can fit 43 months
    if year_str != 'all':
        year_str = f'{year_str}-01-01 {year_stop}-11-01'
    return year_str


def is_error_meme(meme: dict, cat_names: list[str], cwd: str) -> bool:
    errored_json_path = os.path.join(cwd, 'gtrends_data', 'errored_memes.json')
    meme_name = meme['name']

    if os.path.isfile(errored_json_path):
        with open(errored_json_path, "r") as file:
            error_meme_dict = json.load(file)
            for cat_name in cat_names:
                for error_meme in error_meme_dict[cat_name]:
                    if error_meme['name'] == meme_name:
                        return True
            return False
    return False


def update_completed_memes_csv(completed_memes: list[dict], cwd: str):
    completed_json_path = os.path.join(cwd, 'gtrends_data', 'completed_memes.json')
    completed_meme_dict = {'completedMemes': completed_memes}

    with open(completed_json_path, "w") as file:
        json.dump(completed_meme_dict, file, indent='\t')


def check_if_completed(meme, completed_memes) -> bool:
    meme_name = meme['name']

    for completed_meme in completed_memes:
        if completed_meme['name'] == meme_name:
            print(f'Duplicate meme "{meme_name}"')
            return True

    print('meme is not already completed: ' + meme_name)
    return False


def call_trends_api(req: TrendReq, interest_type, times_run=0) -> Union[pd.DataFrame, None]:
    try:
        data = None

        if interest_type == 0:
            data = req.interest_over_time()
        elif interest_type == 1:
            data = req.interest_by_region(resolution='COUNTRY', inc_low_vol=False, inc_geo_code=False)

        print(data)
        if data.empty:
            raise Exception('Dataframe is empty.')

        return data
    except Exception as ex:
        print(f'Got error in request: {ex}')
        #print('Initiating 10 second cooldown...')

        # for s in range(10):
        #     time.sleep(1)

        if times_run == 10:
            print('Maximum retry count of 10 has been reached, returning None.')
            return None

        call_trends_api(req, interest_type, times_run=times_run + 1)


def get_completed_meme_csv(meme_name: str, json_data: dict, curwd: str) -> str:
    cat_names: list[str] = json_data['categories']
    for name in cat_names:
        for meme in json_data[name]:
            if meme_name == meme['name']:
                friendly_name = slugify(meme_name)
                csv_path = os.path.join(curwd, 'gtrends_data', name, f'{friendly_name}.csv')
                return csv_path


def prior_completed_memes(curwd: str) -> list[dict]:
    completed_json_path = os.path.join(curwd, 'gtrends_data', 'completed_memes.json')

    if os.path.isfile(completed_json_path):
        with open(completed_json_path, "r") as file:
            completed_memes = json.load(file)

    return completed_memes['completedMemes']


def save_meme_data(meme: dict, category: str, g_data: pd.DataFrame, gtrends_interest_type: int, curwd: str) -> None:
    friendly_name = slugify(meme["name"])
    csv_path = os.path.join(curwd, 'gtrends_data', category, f'{friendly_name}.csv')
    if gtrends_interest_type == 1:
        csv_path = os.path.join(curwd, 'gtrends_data', 'regional_interest', f'{friendly_name}.csv')

    g_data.to_csv(csv_path, sep='\t')


def save_errored_memes(meme: dict, category: str, curwd: str) -> None:
    errored_json_path = os.path.join(curwd, 'gtrends_data', 'errored_memes.json')
    error_meme_dict = {'erroredMemes': {}}
    error_meme_dict = error_meme_dict['erroredMemes']

    if os.path.isfile(errored_json_path):
        with open(errored_json_path, "r") as file:
            error_meme_dict = json.load(file)

    if category not in error_meme_dict:
        error_meme_dict[category] = []
    error_meme_dict[category].append(meme)

    with open(errored_json_path, "w") as file:
        json.dump(error_meme_dict, file, indent='\t')


def find_instances(meme_name: str, json_data: dict) -> list[dict]:
    instances: list[dict] = []

    for cat_name in json_data['categories']:
        cat = json_data[cat_name]
        for meme in cat:
            if meme['name'] == meme_name:
                instances.append(meme)
                # print('duplicate meme found: ' + meme_name)

    return instances if len(instances) > 1 else None


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


if __name__ == '__main__':
    main()
