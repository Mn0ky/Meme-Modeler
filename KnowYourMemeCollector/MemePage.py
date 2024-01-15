import lxml
import requests
import urllib.parse


class MemePage:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0'}

    def __init__(self, link: str):
        self.gtrends_keywords: list[str] = None
        self.link: str = link
        self.name: str = self.link.split('/')[-1].replace('-', ' ')  # Parse meme name from its URL
        self.year_started: int = -1

    def populate_keywords_and_date(self) -> None:
        html = requests.get(self.link, headers=MemePage.headers)
        parsed_html = lxml.html.fromstring(html.content)

        #'//*[@id="entry_body"]/aside/dl[1]/dd[3]/a/text()'
        # Year meme originated
        self.year_started = parsed_html.xpath('//*[@id="entry_body"]/aside/dl[1]//a[contains(@href, "year")]/text()')
        self.year_started = int(self.year_started[0]) if len(self.year_started) > 0 else -1
        print(self.year_started)

        gtrends_url = parsed_html.xpath('//*[@id="entry_body"]/section/iframe/@data-src')
        gtrends_url = gtrends_url[0] if len(gtrends_url) > 0 else None
        if gtrends_url is None:
            return

        gtrends_url_decoded = urllib.parse.parse_qs(gtrends_url)  # Make trend url readable
        # print(gtrend_url_decoded)
        self.gtrends_keywords = gtrends_url_decoded['q']  # Grab list of phrases to plug into gtrends
        self.gtrends_keywords = self.gtrends_keywords[0].replace("\\'", '').split(',')  # Make list right size
        print(f'For "{self.name}" meme got keywords: {self.gtrends_keywords}')
