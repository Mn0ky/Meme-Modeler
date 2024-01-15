import json
import os
import time
from pytrends.request import TrendReq

curwd = os.path.dirname(os.path.abspath(__file__))
errored_json_path = os.path.join(curwd, 'gtrends_data', 'errored_memes.json')
category_names = ['pop culture reference', 'viral video', 'exploitable', 'catchphrase', 'character']

total_error_memes = 0
error_meme_dict: dict = {}
with open(errored_json_path, "r") as file:
    error_meme_dict: dict = json.load(file)
    cur_meme_index = 0
    total_memes = sum(len(error_meme_dict[cat_name]) for cat_name in category_names)
    #print(f'total error memes before de-duplication: {total_memes}')
    for cat_name in category_names:
        seen_memes = []
        cur_meme_list = error_meme_dict[cat_name]
        for meme in cur_meme_list:
            if meme in seen_memes:
                cur_meme_list.remove(meme)
                continue
            seen_memes.append(meme)
        error_meme_dict[cat_name] = cur_meme_list

    total_error_memes = sum(len(error_meme_dict[cat_name]) for cat_name in category_names)
    print(f'Total memes whose data was unable to be processed: {total_error_memes}')

with open(errored_json_path, "w") as file:
    json.dump(error_meme_dict, file, indent='\t')

json_path = os.path.join(curwd, 'meme_keywords.json')
with open(json_path, "r") as file:
    meme_dict = json.load(file)
    total_memes = sum(len(meme_dict[cat_name]) for cat_name in category_names)
    print(f'Total memes whose data was collected {total_memes - total_error_memes}')
# kw_list = ['hypnotoad', 'all glory to the hypnotoad']
# pytrends = TrendReq(hl='en-US', tz=0, retries=3, backoff_factor=2)
# pytrends.build_payload(kw_list, cat=0, timeframe='all', geo='', gprop='')
# data = pytrends.interest_over_time()
#
# max_kw = data.iloc[:, 1:-1].idxmax(axis=1).iloc[0]  # Gets column with max value between date and ispartial column
#
# print("best keyword is " + max_kw)
# pytrends.build_payload([max_kw], cat=0, timeframe='all', geo='', gprop='')
#
# print(str(pytrends.interest_over_time().to_numpy()))
