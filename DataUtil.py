import os
import time

from pymongo import MongoClient
from alive_progress import alive_bar, alive_it

DEFAULT_MONGO_URI = 'mongodb://localhost:27017'

# These mongo functions are just for when I forget
def get_mongo_client(mongo_uri = DEFAULT_MONGO_URI):
    return MongoClient(mongo_uri)

def get_mongo_db(mongo_client, dbname):
    return mongo_client[dbname]

def get_mongo_collection(dbname, collection_name, mongo_client=None):
    if mongo_client is None:
        return get_mongo_client()[dbname][collection_name]
    else:
        return mongo_client[dbname][collection_name]
    
def insert_one(collection, data):
    collection.insert_one(data)

# note: make sure the line is stripped of trailing new line character
def insert_csv_row(collection, header_values, line):
    collection.insert_one(document_from_values(header_values, line.split(',')))

def document_from_values(header, values):
    result = dict(zip(header, values))
    for key in result:
        if result[key].isdigit():
            result[key] = int(result[key])
        elif '.' in result[key]:
            try:
                result[key] = float(result[key])
            except:
                pass
    return result

def csv_to_mongo(filename, collection_name):
    file_path = f'data/%s' % filename
    collection = get_mongo_collection('finance', collection_name)

    print(f'Reading file: %s' % file_path)
    with open(file_path, 'r') as file:
        header_values = file.readline().strip().split(',')
        for line in alive_it(file.readlines()):
            insert_csv_row(collection, header_values, line.strip())

def get_csv_file_name(ticker, type, resolution, date_from, date_to):
    return f'%s_%s_%s_%s_%s.csv' % (ticker, type, resolution, date_from, date_to)


