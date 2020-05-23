import os
import re
import pandas as pd
from src import settings

def get_data_path(*path_parts):
    '''e.g. get_data_path('in', 'myfile.txt', makedirs=True)
    Will return the absolute path to data/in/myfile.txt
    and create the intermediate folders if makedirs is True
    '''

    ret = os.path.join(settings.DATA_PATH, *path_parts)

    return ret

def read_file(path):
    with open(path, 'rt', encoding='utf-8') as fh:
        ret = fh.read()

    return ret

def read_df_from_titles(path):
    '''returns a pandas dataframe from a titles.txt input file
    dataframe columns:
        title, id, cat, cat1, cat2
    '''
    content = read_file(path)

    titles = re.findall(r'(?m)^\s*(.*?)\s*(\d+)\s*\(([^)]+)\)[^(]*$', content)

    if settings.EXPECTED_TITLE_IDS:
        diff = set(settings.EXPECTED_TITLE_IDS) - set([int(t[1]) for t in titles])
        if diff:
            print(len(titles))
            print(titles[0:2])
            print(sorted(diff))
            exit()

    labels = ['title', 'id', 'cat']
    df = pd.DataFrame.from_records(titles, columns=labels)

    # split multiple categories into cat1 and cat2
    df = df.join(df['cat'].str.split(r'/', 1, expand=True)).rename(columns={0: 'cat1', 1: 'cat2'})

    # bug in the input file: 73 stands for 703, it seems
    df['cat1'].replace({'73': '703'}, inplace=True)

    return df

def split_dataset(df, depth=3, cat_test=2):
    '''
    Shuffles and split the dataset into training and testing samples.
    Uses a new column 'test' = 1|0 to determine if a sample is test or not.

    :param df: dataframe with all titles
    :param depth: depth of the taxonomy for the classification
    :param cat_test: minimum number of test sample per class
    :return: shuffled dataframe with a new column 'test': 1 or 0
    '''
    # shuffle the data
    df = df.sample(frac=1, random_state=settings.SAMPLE_SEED).reset_index(drop=True)
    # create new col with desired cat depth
    df['cat'] = df['cat1'].apply(lambda v: v[:depth])
    # count cats
    vc = df['cat'].value_counts()
    # print(vc)

    # only get cat which have enough samples
    cat_train = cat_test
    vc = vc.where(lambda x: x >= cat_train + cat_test).dropna()
    vc = {k: 2 for k in vc.index}
    # print(vc)

    # split train - test
    df['test'] = 0
    for idx, row in df.iterrows():
        cat = row['cat']
        left = vc.get(cat, 0)
        if left:
            vc[cat] = left - 1
            df.loc[idx, 'test'] = 1

    return df

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

SEVERITY_INFO = 10
SEVERITY_WARNING = 20
SEVERITY_ERROR = 30

def log(message, severity=SEVERITY_INFO):
    severities = {
        SEVERITY_INFO: 'INFO',
        SEVERITY_WARNING: 'WARNING',
        SEVERITY_ERROR: 'ERROR',
    }
    print('[{}] {}'.format(severities[severity], message))
    if severity >= SEVERITY_ERROR:
        exit()

def log_error(message):
    log(message, SEVERITY_ERROR)
