import os

directory = '/home/catherine/stock_data/daily'
outfile_path = '/home/catherine/stock_data/daily.csv'

with open(outfile_path, 'w') as outfile:
    outfile.write('ticker,region,date,open,high,low,close,volume\n')
    for filename in os.listdir(directory):
        #Assume all files are csv
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            firstline = True

            # <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
            for line in file.readlines():
                if firstline:
                    firstline = False
                else:
                    lineitems = line.split(',')
                    writeline = ','.join([
                        filename.split('.')[0],     # ticker
                        filename.split('.')[1],     # region
                        lineitems[2],               # date
                        lineitems[4],               # open
                        lineitems[5],               # high
                        lineitems[6],               # low
                        lineitems[7],               # close
                        lineitems[8]                # volume
                    ])
                    outfile.write(writeline + '\n')

# code for creating a testing data set
# date_start = datetime(2016, 1, 1)
# date_end = datetime(2020, 1, 1)

# data = pd.read_csv('/home/catherine/stock_data/daily.csv')
# data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')

# data = data[(data['date'] >= date_start) & (data['date'] < date_end)]

# # Crude check on data. Assume aapl has full coverage and remove any entry that doesn't have the same num of points
# num_aapl = len(data[data['ticker'] == 'aapl'])
# assert num_aapl > 0

# assets = data['ticker'].unique()

# print(f'%d assets loaded' % len(assets))

# for asset in assets:
#     if len(data[data['ticker'] == asset]) != num_aapl:
#         # print(f'dropping asset %s' % asset)
#         data = data[data['ticker'] != asset]

# data = data.set_index(['region', 'ticker', 'date'])
# data.to_csv('/home/catherine/stock_data/test_data.csv', )
# print()
