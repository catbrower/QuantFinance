import requests
import logging
from itertools import product
import pandas as pd
# from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium_stealth import stealth

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

binary_location = '/usr/bin/geckodriver'
valid_symbols = 'abcdefghijklmnopqrstuvwxyz.'
url_stocks = 'https://www.nasdaq.com/market-activity/stocks'
data_file_location = 'DataScrapers/nasdaq_tickers.csv'

options = Options()
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument('-headless=new')
# options.binary_location = binary_location
# driver = webdriver.Firefox(service=Service(executable_path=binary_location, options=options))

def get_tickers():
    driver = webdriver.Chrome(options)
    data = pd.read_csv(data_file_location)
    data = data.set_index('id')

    stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
        )

    for length in range(1, 6):
        for symbol in product(valid_symbols, repeat=length):
            symbol = "".join(symbol)

            if symbol[0] == '.' or symbol[-1] == '.' or symbol in data['ticker'].tolist():
                continue
            
            url = f'{url_stocks}/{symbol}'
            driver.get(url)
            elems = driver.find_elements(By.CLASS_NAME, 'alert__heading')
            if len(elems) == 1:
                if elems[0].text == f'{symbol.upper()} is currently not trading.':
                    data.loc[max(data.index) + 1] = [symbol, 0]
                else:
                    logger.error(f'Unexpected condition for symbol: {symbol}. Unexpected header text.')
            elif len(elems) == 0:
                data.loc[max(data.index) + 1] = [symbol, 1]
            else:
                logger.error(f'Unexpected condition for symbol: {symbol}. Unexpected number of header results.')

            data.to_csv(data_file_location)

get_tickers()




