from datetime import datetime
from io import StringIO
from os import path, makedirs
from timeit import default_timer as timer
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

# ReadMe: http://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/cotacoes-historicas/
# Source: http://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/series-historicas/
# Layout: http://www.b3.com.br/data/files/33/67/B9/50/D84057102C784E47AC094EA8/SeriesHistoricas_Layout.pdf
b3_cotahist_urls = {
    '2015': 'https://web.archive.org/web/20170323062413if_/http://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A2015.ZIP',
    '2016': 'https://web.archive.org/web/20170323062453if_/http://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A2016.ZIP',
    '2017': 'https://web.archive.org/web/20210802173802if_/http://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A2017.ZIP',
    '2018': 'https://web.archive.org/web/20210802182640if_/http://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A2018.ZIP',
    '2019': 'https://web.archive.org/web/20210802184310if_/http://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A2019.ZIP'
}

# Antes de 2018-03-26 'B3SA3' era listada como 'BVMF3'
top_ten_stocks = {
    '2017': ['ITUB4', 'BBDC4', 'ABEV3', 'PETR4', 'VALE3', 'BRFS3', 'BBAS3', 'ITSA4', 'BVMF3', 'UGPA3'],
    '2018': ['ITUB4', 'VALE3', 'BBDC4', 'ABEV3', 'PETR4', 'BVMF3', 'B3SA3', 'ITSA4', 'BBAS3', 'UGPA3', 'BRFS3'],
    '2019': ['ITUB4', 'VALE3', 'BBDC4', 'PETR4', 'ABEV3', 'BBAS3', 'B3SA3', 'ITSA4', 'LREN3', 'UGPA3']
}

test_years = {
    '2015': ['2017'],
    '2016': ['2017', '2018'],
    '2017': ['2017', '2018', '2019'],
    '2018': ['2018', '2019'],
    '2019': ['2019'],
}


def download_file(url, out_filename):
    if not path.isfile(out_filename):
        makedirs(path.dirname(out_filename), exist_ok=True)

        with open(out_filename, 'wb') as out_zip:
            return out_zip.write(urlopen(url).read())


def read_zipfile(zip_filename, filename):
    with ZipFile(zip_filename) as zip_file:
        return zip_file.open(filename).read().decode('ascii')


def read_b3_cotahist_zips(b3_urls):
    txts = {}

    for year, url in b3_urls.items():
        zip_filename = f"zips/COTAHIST_A{year}.ZIP"
        txt_filename = f"COTAHIST_A{year}.TXT"

        download_file(url, zip_filename)
        txts[year] = read_zipfile(zip_filename, txt_filename)

    return txts


def gen_b3_cotahist_csv(b3_urls, csv_filename, include_list):
    b3_cotahist_txts = read_b3_cotahist_zips(b3_urls)

    with open(csv_filename, 'w') as writefile:
        writefile.write('Symbol,Date,Open,High,Low,Close,Volume\n')

        for year, b3_cotahist_txt in b3_cotahist_txts.items():
            for ln, line in enumerate(StringIO(b3_cotahist_txt)):
                if ln > 0:
                    code = line[12:24].strip()

                    if year in include_list and code in include_list[year]:
                        date = datetime.strptime(line[2:10], '%Y%m%d').strftime('%Y-%m-%d')
                        price_open = int(line[56:69]) / 100
                        price_high = int(line[69:82]) / 100
                        price_low = int(line[82:95]) / 100
                        price_close = int(line[108:121]) / 100
                        volume = int(int(line[170:188]) / 100)

                        writefile.write(
                            f"{code},{date},{price_open:.2f},{price_high:.2f},{price_low:.2f},{price_close:.2f},{volume}\n")


def main():
    csv_filename = 'data/B3_COTAHIST.csv'
    include_list = {year: set([s for y in ys for s in top_ten_stocks[y]]) for year, ys in test_years.items()}

    start = timer()
    gen_b3_cotahist_csv(b3_cotahist_urls, csv_filename, include_list)
    print("Parse time: {:.3f}s".format(timer() - start))

    start = timer()
    df = pd.read_csv(csv_filename, parse_dates=True, index_col='Date')
    print("Read time: {:.3f}s\n".format(timer() - start))

    print('Pandas DataFrame:')
    print(df)


if __name__ == '__main__':
    main()
