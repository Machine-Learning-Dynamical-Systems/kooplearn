from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import os
import shutil


def download_and_unzip(url, extract_to='_tmp'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

def get_file_list(download_path):
    data_files = []
    for root, _, files in os.walk(download_path, topdown=False):
        for name in files:
            data_files.append(os.path.join(root, name))
    return data_files

def prepare_dataframe(file_path):
    df = pd.read_csv(file_path).drop(["No"], axis = 1)
    df['wd'] = df['wd'].replace(to_replace=__wind_directions_to_degrees)
    #df['uid'] = df['year'].astype(str) + "_" +  df['month'].astype(str) + "_" + df['day'].astype(str) + "_" + df['hour'].astype(str)
    df['uid'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('uid')
    station_name = df['station'].unique()
    assert len(station_name) == 1
    station_name = station_name[0]
    df = df.drop(['month', 'day', 'hour', 'year', 'station'], axis = 1)
    name_mapper = dict()
    for col in df.columns:
        name_mapper[col] = col + "_" + station_name
    df = df.rename(columns=name_mapper)
    return df

__wind_directions_to_degrees = {
    'N': 0.0,
    'NNE': 22.5,
    'NE': 45.0,
    'ENE': 67.5,
    'E': 90.0,
    'ESE': 112.5,
    'SE': 135.0,
    'SSE': 157.5,
    'S': 180.0,
    'SSW': 202.5,
    'SW': 225.0,
    'WSW': 247.5,
    'W': 270.0,
    'WNW': 292.5,
    'NW': 315.0,
    'NNW': 337.5
}

if __name__ == "__main__":
    download_path = "_tmp"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip"
    download_and_unzip(url, download_path)
    file_list = get_file_list(download_path)
    dfs = [prepare_dataframe(f) for f in file_list]
    df = pd.concat(dfs, axis=1)
    if not os.path.exists('data'):  
        os.makedirs('data')
    df.to_pickle("data/full_dataframe")
    shutil.rmtree(download_path)