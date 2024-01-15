import json
import os
import re
import shutil
import time
import unicodedata
import urllib.parse
from typing import Union

import pandas as pd
from pytrends.request import TrendReq


def main() -> None:
    pytrends = TrendReq(hl='en-US', tz=0, retries=3, backoff_factor=2)
    curwd = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(curwd, 'meme_keywords.json')

    with open(json_path) as f:
        json_data = json.load(f)

    cat_names: list[str] = json_data['categories']

    # Remove duplicates, decode any url encoded meme names, and get the data
    completed_memes: list[dict] = prior_completed_memes(json_data, curwd)
    for name in cat_names:
        cat_memes: list[dict] = json_data[name]

        for meme in cat_memes:
            meme_name = meme['name']

            if '%' in meme_name:
                decoded_meme_name = urllib.parse.unquote(meme_name)
                meme['name'] = decoded_meme_name
                print(f'decoded "{meme_name}" into "{decoded_meme_name}"')

            if handle_completed_meme(meme, name, json_data, completed_memes, curwd):
                print(f'Is completed meme "{meme_name}", skipping...')
                continue

            if is_error_meme(meme, cat_names, curwd):
                print(f'Is error meme "{meme_name}", skipping...')
                continue

            print('Getting gdata for meme ' + meme['name'])
            g_data = get_meme_data(meme, pytrends)
            if g_data is None:
                print(f'Unable to get data for {meme["name"]}, adding to errors list.')
                save_errored_memes(meme, name, curwd)
                continue

            save_meme_data(meme, name, g_data, curwd)
            completed_memes.append(meme)


def get_meme_data(meme: dict, pytrends: TrendReq) -> Union[pd.DataFrame, None]:
    kw_list = meme['keywords']  # ['Giorgio A. Tsoukalos', 'Giorgio Tsoukalos']
    year = meme['year']
    year_str = str(year - 2)  # Start 2 years before meme "got popular"
    year_stop = '2023'

    if year <= 2004:
        year_str = 'all'
    elif year > 2021:
        year_str = str(2019)  # Make sure period can fit 43 months
    if year_str != 'all':
        year_str = f'{year_str}-01-01 {year_stop}-11-01'

    if kw_list is None:  # For memes without keywords
        return None
    if len(kw_list) == 1:
        pytrends.build_payload(kw_list, cat=0, timeframe=year_str, geo='', gprop='')
        data = call_trends_api(pytrends)
        print(f'Successfully got data for "{meme["name"]}"')
        return data

    # For memes with multiple keywords
    pytrends.build_payload(kw_list, cat=0, timeframe=year_str, geo='', gprop='')
    data = call_trends_api(pytrends)
    if data is None:
        return None
    max_kw = data.iloc[:, 1:-1].idxmax(axis=1).iloc[0]  # Gets column with max value between date and ispartial column
    pytrends.build_payload([max_kw], cat=0, timeframe=year_str, geo='', gprop='')
    data = call_trends_api(pytrends)  # GTrends data from "most popular" keyword

    return data


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


def handle_completed_meme(meme, cur_cat, json_data, completed_memes, cwd) -> bool:
    # For memes in multiple categories
    meme_name = meme['name']
    #print(completed_memes)
    for completed_meme in completed_memes:
        #print('comp meme: ' + completed_meme['name'])
        if completed_meme['name'] == meme_name:
            print(f'Duplicate meme "{meme_name}"')
            completed_csv = get_completed_meme_csv(meme_name, json_data, cwd)
            dest = os.path.join(cwd, 'gtrends_data', cur_cat, os.path.basename(completed_csv))
            try:
                shutil.copy(completed_csv, dest)
                print(f'Copied meme duplicate "{meme_name}"')
            except shutil.SameFileError:
                pass  # Ignore double copies

            return True
    print('meme is not already completed: ' + meme_name)
    return False


def call_trends_api(req: TrendReq, times_run=0) -> Union[pd.DataFrame, None]:
    try:
        data = req.interest_over_time()
        print(data)
        if data.empty:
            raise Exception('Dataframe is empty.')

        return data
    except Exception as ex:
        print(f'Got error in request: {ex}')
        #print('Initiating 10 second cooldown...')

        # for s in range(10):
        #     time.sleep(1)

        if times_run == 20:
            print('Maximum retry count of 20 has been reached, returning None.')
            return None

        call_trends_api(req, times_run=times_run + 1)


def get_completed_meme_csv(meme_name: str, json_data: dict, curwd: str) -> str:
    cat_names: list[str] = json_data['categories']
    for name in cat_names:
        for meme in json_data[name]:
            if meme_name == meme['name']:
                friendly_name = slugify(meme_name)
                csv_path = os.path.join(curwd, 'gtrends_data', name, f'{friendly_name}.csv')
                return csv_path


def prior_completed_memes(json_data: dict, curwd: str) -> list[dict]:
    comp_memes: list[dict] = []

    for cat in json_data['categories']:
        for meme in json_data[cat]:
            meme_fname = slugify(meme['name'])
            csv_path = os.path.join(curwd, 'gtrends_data', cat, f'{meme_fname}.csv')
            if os.path.isfile(csv_path):
                print(f'Adding "{meme["name"]}" to completed list...')
                comp_memes.append(meme)

    return comp_memes


def save_meme_data(meme: dict, category: str, g_data: pd.DataFrame, curwd: str) -> None:
    friendly_name = slugify(meme["name"])
    csv_path = os.path.join(curwd, 'gtrends_data', category, f'{friendly_name}.csv')
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


def slugify(value, allow_unicode=True):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def find_instances(meme_name: str, json_data: dict) -> list[dict]:
    instances: list[dict] = []

    for cat_name in json_data['categories']:
        cat = json_data[cat_name]
        for meme in cat:
            if meme['name'] == meme_name:
                instances.append(meme)
                # print('duplicate meme found: ' + meme_name)

    return instances if len(instances) > 1 else None


if __name__ == '__main__':
    main()
