import lxml
import requests
import time

from MemePage import MemePage


class CategoryPage:

    def __init__(self, link: str, name: str):
        self.link: str = link
        self.name = name
        self.meme_pages: list[MemePage]
        self.total_pages: int = -1
        self.sort_filter: str = '?sort=views'

    def process_meme_pages(self):
        self._populate_meme_pages()
        self._iterate_meme_pages()

    def _populate_meme_pages(self):
        html = requests.get(self.link + self.sort_filter, headers=MemePage.headers)
        parsed_html = lxml.html.fromstring(html.content)

        meme_links = parsed_html.xpath('//*[@id="entries_list"]/table[1]//a/@href')
        # print(meme_links)

        # Category's total # of pages
        self.total_pages = int(parsed_html.xpath('//*[@id="entries"]/aside/div//a/text()')[-2])
        self.meme_pages = []
        print(f'There are {self.total_pages} total pages of memes for "{self.name}" category.')
        print(f'Processed page #1 for "{self.name}" category.')
        # print(num_pages)

        for i in range(2, self.total_pages + 1):
            time.sleep(5)  # Delay to avoid IP ban
            next_page_link = f'{self.link}/page/{i}{self.sort_filter}'
            html = requests.get(next_page_link, headers=MemePage.headers)
            parsed_html = lxml.html.fromstring(html.content)
            next_meme_links = parsed_html.xpath('//*[@id="entries_list"]/table[1]//a/@href')
            #print(f'next meme links of size {len(next_meme_links)}')
            meme_links += next_meme_links
            print(f'Processed page #{i} for "{self.name}" category.')

        meme_links = list(set(meme_links))  # Remove duplicates
        self.meme_pages = [MemePage(link) for link in meme_links]

    def _iterate_meme_pages(self):
        for page in self.meme_pages:
            page.populate_keywords_and_date()
            time.sleep(5)
