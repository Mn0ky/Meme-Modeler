import requests
import lxml.html
import json
import os
import re

from CategoryPage import CategoryPage
from MemePage import MemePage


def main() -> None:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0'}
    curwd = os.path.dirname(os.path.abspath(__file__))
    site = 'https://knowyourmeme.com'

    # More categories to add later (didn't have 450 min.): parody
    category_names = ['pop culture reference', 'viral video', 'exploitable', 'catchphrase', 'character']
    categories = []

    html = requests.get(site + '/categories', headers=headers)
    parsed_html = lxml.html.fromstring(html.content)

    category_elems = parsed_html.xpath('//*[@id="categories-list"]/a[contains(@href,"types")]')

    for elem in category_elems:
        category_name: str = elem.text_content().lower()
        category_name = re.sub(r'[0-9]', '', category_name).replace(' ,', '')  # Remove number plus commas from the end
        #print(category_name)

        if category_name in category_names:
            category_link: str = elem.attrib['href']
            category_link = category_link.replace('?status=all', '')  # Only confirmed, popular memes
            category = CategoryPage(site + category_link, category_name)
            categories.append(category)

    data_dict = {'categories': category_names}

    for category_name in category_names:
        data_dict[category_name]: list[dict] = []

    for category in categories:
        category.process_meme_pages()
        category_section: list[dict] = data_dict[category.name]

        for meme_page in category.meme_pages:
            category_section.append({
                'name': meme_page.name,
                'year': meme_page.year_started,
                'keywords': meme_page.gtrends_keywords,
            })

        print(f'Finished processing "{category.name}"! Updated data dict...')
        print(data_dict)

    json_path = os.path.join(curwd, 'meme_keywords.json')
    with open(json_path, "w") as file:
        json.dump(data_dict, file, indent='\t')

    print('Finished processing, data has been dumped to meme_keywords.json')


if __name__ == '__main__':
    main()
