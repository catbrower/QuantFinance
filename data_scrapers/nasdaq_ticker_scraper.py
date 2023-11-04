import scrapy
import sqlite3
import itertools

class UrlLoggingPipeline:
    def __init__(self):
        self.conn = sqlite3.connect('url_logs.db')
        self.cursor = self.conn.cursor()
        self.existing_symbols = set()  # Keep track of existing symbols

        # Load existing symbols from the database
        self.cursor.execute('SELECT DISTINCT symbol FROM url_logs')
        for row in self.cursor.fetchall():
            self.existing_symbols.add(row[0])

    def process_item(self, item, spider):
        symbol = item['symbol']
        status_code = item['status_code']
        url = item['url']

        # Check if the symbol already exists in the database
        if symbol in self.existing_symbols:
            self.log_skipped_item(symbol)
        else:
            # Insert the data into the database
            self.cursor.execute('''
                INSERT INTO url_logs (url, symbol, status_code) VALUES (?, ?, ?)
            ''', (url, symbol, status_code))
            self.conn.commit()

        return item

    def log_skipped_item(self, symbol):
        # Log the skipped symbol to indicate that it's already in the database
        print(f'Skipped symbol: {symbol} (Already recorded)')

    def close_spider(self, spider):
        # Close the database connection when the spider is finished
        self.conn.close()

class NasdaqTickerSpider(scrapy.Spider):
    name = 'your_spider_name'
    allowed_domains = ['nasdaq.com']

    def __init__(self, *args, **kwargs):
        super(NasdaqTickerSpider, self).__init__(*args, **kwargs)
        self.start_urls = self.generate_start_urls()

    def generate_start_urls(self):
        # Define the characters you want to use
        characters = 'abcdefghijklmnopqrstuvwxyz'
        urls = []
        
        # Generate all combinations of six characters
        symbol_combinations = [''.join(combination) for combination in itertools.product(characters, repeat=6) if combination[0] != '.' and combination[-1] != '.']
        
        for symbol in symbol_combinations:
            if symbol not in self.existing_symbols:  # Check if the symbol exists in the database
                url = f'https://www.nasdaq.com/market-activity/stocks/{symbol}/dividend-history'
                urls.append(url)
        
        return urls

    def parse(self, response):
        symbol = response.meta['symbol']
        status_code = response.status
        url = response.url

        your_item = NasdaqTicker()
        your_item['title'] = response.xpath('your-title-xpath').get()
        your_item['link'] = response.xpath('your-link-xpath').get()
        your_item['symbol'] = symbol
        your_item['status_code'] = status_code
        your_item['url'] = url

        yield your_item

# Define the data item
class NasdaqTicker(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    symbol = scrapy.Field()
    status_code = scrapy.Field()
    url = scrapy.Field()