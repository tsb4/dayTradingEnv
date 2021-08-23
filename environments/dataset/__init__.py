from os import path

import pandas as pd

from .b3 import top_ten_stocks

B3_STOCKS = pd.read_csv(path.join(path.dirname(path.abspath(__file__)), 'data', 'B3_COTAHIST.csv'),
                        parse_dates=True, index_col='Date')
