import os
import re
import pandas as pd
import settings

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

def split_dataset(df, depth=3, cat_train=2, cat_test=2, cat_valid=0):
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

    if 0:
        # save number of samples per class
        vc.to_csv(get_data_path('out', 'cat-{}.csv'.format(depth)))

    # only get cat which have enough samples
    vc = vc.where(lambda x: x >= (cat_train + cat_test + cat_valid)).dropna()
    # print(vc, len(vc))
    vc = {k: cat_test for k in vc.index}
    # exit()
    # print(vc)

    # split train - test
    df['test'] = 0
    for idx, row in df.iterrows():
        cat = row['cat']
        left = vc.get(cat, 0)
        if left:
            vc[cat] = left - 1
            df.loc[idx, 'test'] = 1 if (left > cat_valid) else 2

    # df = df.append(get_class_titles(depth))

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

def extract_transcripts_from_pdfs():

    fh = open(get_data_path('out', 'transcripts.txt'), 'wb')

    import textract
    from pathlib import Path
    paths = list(Path(get_data_path('in', 'pdfs')).rglob("*TEXT*.pdf"))

    paths = [str(p) for p in paths]
    for p in sorted(paths, key=lambda p: os.path.basename(p)):
        print(p)
        fh.write(('\n\nNEWFILE {}\n\n'.format(os.path.basename(p))).encode('utf-8'))

        content = textract.process(p, method='tesseract', language='eng')

        fh.write(content)

    fh.close()

def get_class_titles(depth=3):

    classes = {}

    content = read_file(get_data_path('in', 'classes.txt'))

    for cls in re.findall(r'(?m)^(\d+)\.?\s+(.+)$', content):
        cls_num = cls[0]
        cls = {
            'title': cls[1],
            'titlen': re.sub(r'\W', ' ', cls[1].lower()),
            'cat1': cls_num,
            'cat2': '',
            'cat': cls_num,
            'id': 'cls_'+cls_num,
            'test': 0,
        }
        # print(cls)
        classes[cls_num] = cls

    data = []
    for cls in classes.values():
        num = cls['cat']
        if len(num) == depth:
            data.append(cls)
            while num:
                num = num[:-1]
                parent = classes.get(num)
                if parent:
                    cls['title'] += ' ' + parent['title']
                    cls['titlen'] += ' ' + parent['titlen']
            # print(cls)

    ret = pd.DataFrame(data)

    return ret
