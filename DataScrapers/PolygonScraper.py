import sys
import ray
import time
import math
import random
import logging
import pymongo
import requests
import functools
import numpy as np
import concurrent.futures
from threading import Thread, Lock
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

NUM_THREADS = 12
url_base = 'https://api.polygon.io/v3/reference'
api_key = 'PrCJ1R_Sa_jfqIzP_un7pjwsVcS_TTd5m_vGs1'
asset_classes = ['stocks', 'options', 'crypto', 'fx', 'indices']

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['polygon']

# Thread vars
global ticker_counter
ticker_counter = 0

# https://www.digitalocean.com/community/tutorials/how-to-use-threadpoolexecutor-in-python-3
class PolygonTickerDetails():
    def __init__(self):
        self.counter = 0
        self._lock = Lock()
        failed = []

    def log_status(self, num_tickers):
        count = 0
        with tqdm(total=num_tickers) as pbar:
            while count < num_tickers:
                if self.counter > count:
                    with self._lock:
                        pbar.update(self.counter - count)
                        count = self.counter
                time.sleep(0.1)
        print('')

    def get_data(self, all_tickers):
        status_monitor = Thread(target=self.log_status, args=[len(all_tickers)])
        status_monitor.start()

        chunks = np.array_split(all_tickers, NUM_THREADS)

        with concurrent.futures.ThreadPoolExecutor(NUM_THREADS) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self.fetch_ticker_details, chunk))

    # Fetch ticker details
    def fetch_ticker_details(self, tickers):
        next_update_time = time.time() + random.random() * 5
        for ticker in tickers:
            data_does_not_exist = db['tickers'].find_one({'ticker': ticker.upper()}) is None

            if data_does_not_exist:
                url = polygon_url(f'/tickers/{ticker.upper()}')
                response = requests.get(url)
                if response.status_code == 200:
                    json = response.json()
                    insert_if_not_exists(json['results'], 'tickers')
                else:
                    self.failed.append(ticker)

            with self._lock:
                self.counter += 1

        for ticker in self.failed:
            logger.error(f'Failed to fetch details for ticker {ticker}')

def polygon_url(endpoint, params={}):
    params['apiKey'] = api_key
    params_str = '&'.join([f'{key}={params[key]}' for key in params])
    return f'{url_base}{endpoint}?{params_str}'

def insert_if_not_exists(document, collection):
    existing_document = db[collection].find_one(document)
    if existing_document is None:
        db[collection].insert_one(document)

def insert_all(items, collection):
    for item in items:
        insert_if_not_exists(item, collection)

# Get exchanges
def fetch_exchanges():
    # response = requests.get(f'{url_base}/exchanges?asset_class={asset_class}&apiKey={api_key}')
    response = requests.get(polygon_url('/exchanges'))
    if response.status_code == 200:
        insert_all(response.json()['results'], 'exchanges')
    else:
        logger.warning(f'Could not fetch exchenges')

# Get all tickers
def fetch_tickers():
    logger.info('Fetching ticker data')
    result = []
    with tqdm() as pbar:
        response = requests.get(polygon_url('/tickers'))
        if response.status_code == 200:
            json = response.json()
            result += json['results']

            while 'next_url' in json:
                response = requests.get(f'{json["next_url"]}&apiKey={api_key}')
                if response.status_code == 200:
                    json = response.json()
                    result += json['results']
                    pbar.update()
                else:
                    logger.error(f'Failed to get:\n\t{json["next_url"]}\nReceived status code: {response.status_code}')
        else:
            logger.error(f'Failed to fetch tickers')
            sys.exit(1)

    return [x['ticker'] for x in result]
        
def fetch_ticker_types():
    response = requests.get(polygon_url('/tickers/types'))
    if response.status_code == 200:
        insert_all(response.json()['results'], 'ticker_types')
    else:
        logger.warning('Could not fetch ticker types')

def fetch_stock_splits(url):
    response = requests.get(url)
    if response.status_code == 200:
        json = response.json()
        insert_all(json['results'], 'stock_splits')
        if json['next_url'] is not None:
            fetch_stock_splits(f'{json["next_url"]}&apiKey={api_key}')
    else:
        logger.warning(f'Could not fetch stock splits for url: {url}')

def fetch_dividends(url):
    response = requests.get(url)
    if response.status_code == 200:
        json = response.json()
        insert_all(json['results'], 'dividends')
        if json['next_url'] is not None:
            fetch_dividends(f'{json["next_url"]}&apiKey={api_key}')
    else:
        logger.warning(f'Could not fetch stock splits for url: {url}')

# TODO polygon api is kind of odd
def fetch_financials(url):
    pass

def fetch_ticker_details():
    # Fetch ticker dretails
    num_tickers_at_start = db['tickers'].count_documents({})
    tickers = fetch_tickers()
    logger.info(f'Found {len(tickers)} tickers')
    tickerDetails = PolygonTickerDetails()
    tickerDetails.get_data(tickers)
    logger.info(f'Added {db["tickers"].count_documents({}) - num_tickers_at_start} new tickers')

# fetch_ticker_details(tickers)

logger.info('Fetching reference data')


logger.info('Fetching agg')

# reference_data_steps = [
#     fetch_exchanges,
#     # lambda: fetch_tickers(polygon_url('/tickers')),
#     fetch_ticker_types,
#     lambda: fetch_stock_splits(polygon_url('/splits')),
#     lambda: fetch_dividends(polygon_url('/dividends'))
# ]

# for step in tqdm(reference_data_steps):
#     step()





