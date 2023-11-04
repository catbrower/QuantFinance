import sqlite3
import pytest
from data_scrapers import UrlLoggingPipeline, nasdaq_ticker_scraper

# Define a test database for SQLite in memory
TEST_DATABASE_NAME = ':memory:'

# Test the Spider
class TestYourSpider:
    @pytest.fixture(autouse=True)
    def setup_spider(self):
        self.spider = nasdaq_ticker_scraper()

    def test_generate_start_urls(self):
        urls = self.spider.generate_start_urls()
        assert isinstance(urls, list)
        assert len(urls) > 0

    # Add more spider-related tests here as needed

# Test the UrlLoggingPipeline
class TestUrlLoggingPipeline:
    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        self.pipeline = UrlLoggingPipeline()
        self.conn = sqlite3.connect(TEST_DATABASE_NAME)
        self.cursor = self.conn.cursor()

    def test_log_skipped_item(self, capsys):
        symbol = 'AAPL'
        self.pipeline.log_skipped_item(symbol)
        captured = capsys.readouterr()
        assert f'Skipped symbol: {symbol}' in captured.out

    # Add more pipeline-related tests here as needed

    def teardown_method(self):
        self.conn.close()

# Define a test database to be used with the pipeline
@pytest.fixture(scope='session')
def test_database():
    conn = sqlite3.connect(TEST_DATABASE_NAME)
    cursor = conn.cursor()

    # Create a table for testing
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS url_logs (
            id INTEGER PRIMARY KEY,
            url TEXT,
            symbol TEXT,
            status_code INTEGER
        )
    ''')

    conn.commit()
    yield conn

    # Clean up the test database after the tests
    conn.close()