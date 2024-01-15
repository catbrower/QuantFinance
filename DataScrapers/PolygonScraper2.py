import sys
import time
import logging
import hashlib
import pymongo
import requests
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
from threading import Thread, Lock
from tqdm import tqdm
from polygon import RESTClient

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

NUM_THREADS = 12
years_of_history = 5
url_base = 'https://api.polygon.io'
api_key = 'PrCJ1R_Sa_jfqIzP_un7pjwsVcS_TTd5m_vGs1'

db_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = db_client['polygon']
api_client = RESTClient(api_key=api_key)

# Update flags
update_tickers = False
update_ticker_details = False
update_aggregates = True

def weekDayGenerator(startDate, endDate):
    currentDate = startDate
    while currentDate < endDate:
        if currentDate.weekday() < 5:
            yield currentDate
        currentDate = currentDate + timedelta(days=1)

def insert_if_not_exists(document, collection):
    existing_document = db[collection].find_one(document)
    if existing_document is None:
        db[collection].insert_one(document)

def insert_all(items, collection):
    for item in items:
        insert_if_not_exists(item, collection)

def polygon_url(endpoint, params={}, version='v3'):
    params['apiKey'] = api_key
    params_str = '&'.join([f'{key}={params[key]}' for key in params])
    return f'{url_base}/{version}/{endpoint}?{params_str}'

def polygon_aggregates_url(ticker, dateFrom, dateTo, timespan='second', multiplier='1'):
    dateFormat = '%Y-%m-%d'
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': '50000'
    }
    url = f'aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/{dateFrom.strftime(dateFormat)}/{dateTo.strftime(dateFormat)}'
    return polygon_url(url, version='v2', params=params)

class PolygonTickerDetails():
    def __init__(self):
        self.counter = 0
        self._lock = Lock()
        self.failed = []

    def log_status(self, num_tickers):
        count = 0
        with tqdm(total=num_tickers) as pbar:
            while count < num_tickers:
                if self.counter > count:
                    with self._lock:
                        pbar.update(self.counter - count)
                        count = self.counter
                time.sleep(0.1)

    def get_data(self):
        all_tickers = [x['ticker'] for x in db['tickers'].find({})]

        status_monitor = Thread(target=self.log_status, args=[len(all_tickers)])
        status_monitor.start()

        chunks = np.array_split(all_tickers, NUM_THREADS)

        with concurrent.futures.ThreadPoolExecutor(NUM_THREADS) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self.fetch_ticker_details, chunk))

    # Fetch ticker details
    def fetch_ticker_details(self, tickers):
        for ticker in tickers:
            data_does_not_exist = db['tickerDetails'].find_one({'ticker': ticker.upper()}) is None

            if data_does_not_exist:
                url = polygon_url(f'reference/tickers/{ticker.upper()}')
                response = requests.get(url)
                if response.status_code == 200:
                    json = response.json()
                    db['tickerDetails'].insert_one(json['results'])
                    # insert_if_not_exists(json['results'], 'tickerDetails')
                else:
                    self.failed.append(ticker)

            with self._lock:
                self.counter += 1

        for ticker in self.failed:
            logger.error(f'Failed to fetch details for ticker {ticker}')

class PolygonAggregateGetter():
    def __init__(self):
        self.counter = 0
        self._lock = Lock()
        self.failed = []
        self.shouldLogStatus = True

    def log_status(self, num_tickers):
        count = 0
        with tqdm(total=num_tickers) as pbar:
            while count < num_tickers and self.shouldLogStatus:
                if self.counter > count:
                    with self._lock:
                        pbar.update(self.counter - count)
                        count = self.counter
                time.sleep(0.1)

    def get_data(self):
        all_tickers = [x['ticker'] for x in db['tickers'].find({})]
        # all_tickers = ['SPY']

        status_monitor = Thread(target=self.log_status, args=[len(all_tickers)])
        status_monitor.start()

        chunks = np.array_split(all_tickers, NUM_THREADS)

        with concurrent.futures.ThreadPoolExecutor(NUM_THREADS) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self.fetch_aggregates, chunk))

        self.shouldLogStatus = False
        status_monitor.join()

    # Fetch ticker details
    def fetch_aggregates(self, tickers):
        if tickers is None:
            print('Tickers is none')
            return
        
        endDate = datetime(datetime.now().year, datetime.now().month, datetime.now().day)
        startDate = datetime(datetime.now().year - years_of_history, datetime.now().month, datetime.now().day)
        for ticker in tickers:
            try:
                if ticker is None:
                    print('Ticker is None')
                    continue

                tickerDetail = db['tickerDetails'].find_one({'ticker': ticker})
                if tickerDetail is not None and 'list_date' in tickerDetail:
                    listDate = datetime.strptime(tickerDetail['list_date'], '%Y-%m-%d')
                    if listDate > startDate:
                        startDate = listDate

                if startDate is None or endDate is None:
                    print('Encounted None type date')
                    continue

                for day in weekDayGenerator(startDate, endDate):
                    # timestamp = day.timestamp() * 1000
                    # data_does_not_exist = db['aggregates'].find_one({'ticker': ticker.upper(), 'timestamp': {'$gte': timestamp, '$lt': timestamp + 8.64e7}}) is None
                    # print(day)
                    # if data_does_not_exist:
                    # try:
                    url = polygon_aggregates_url(ticker, day, day)
                    response = requests.get(url)
                    if response.status_code == 200:
                        json = response.json()
                #         # insert_all(json['results'], 'aggregates')
                        if 'results' in json:
                            for index, item in enumerate(json['results']):
                                item['ticker'] = ticker
                                # item['_id'] = hashlib.md5(''.join([str(item[key]) for key in item]).encode('utf-8')).hexdigest()
                                json['results'][index] = item
                            
                            try:
                                db['aggregates'].insert_many(json['results'])
                            except Exception as err:
                                print('DB insert failed')
                                print(err)
                        else:
                            self.failed.append(ticker)

                with self._lock:
                    self.counter += 1

                for ticker in self.failed:
                    logger.error(f'Failed to fetch details for ticker {ticker}')

            except Exception as err:
                print(f'Processing ticker {ticker} failed on range {startDate} - {endDate}')
                print(err)

        print('done')

if update_tickers:
    for x in tqdm(api_client.list_tickers()):
        insert_if_not_exists(x.__dict__, 'tickers')

if update_ticker_details:
    tickerDetailsGetter = PolygonTickerDetails()
    tickerDetailsGetter.get_data()

if update_aggregates:
    aggregateDataGetter = PolygonAggregateGetter()
    aggregateDataGetter.get_data()