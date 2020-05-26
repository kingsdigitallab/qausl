import os
import re
import pandas as pd
import settings
import fasttext
from _csv import QUOTE_NONE

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
    vc = {k: cat_test + cat_valid for k in vc.index}

    # split 0:train - 1:test - 2:valid
    df['test'] = -1
    for idx, row in df.iterrows():
        cat = row['cat']
        left = vc.get(cat, 0)
        if left > 0:
            df.loc[idx, 'test'] = 1 if (left > cat_valid) else 2
        else:
            if left > - settings.TRAIN_PER_CLASS_MAX:
                df.loc[idx, 'test'] = 0
        vc[cat] = left - 1

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
    import time

    severities = {
        SEVERITY_INFO: 'INFO',
        SEVERITY_WARNING: 'WARNING',
        SEVERITY_ERROR: 'ERROR',
    }
    full_message = '[{}] {}'.format(severities[severity], message)
    print(full_message)

    with open(get_data_path('out', 'experiments.log'), 'at') as fh:
        fh.write('[{}] {}'.format(time.strftime("%Y-%m-%d %H:%M"), full_message+'\n'))

    if severity >= SEVERITY_ERROR:
        exit()


def log_error(message):
    log(message, SEVERITY_ERROR)


def extract_transcripts_from_pdfs():
    fh = open(settings.TRANSCRIPTS_PATH, 'wb')

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
            'id': 'cls_' + cls_num,
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

def save_ft_sets(df, titles_out_path):
    # prepare the label for fasttext format
    df['label'] = df['cat'].apply(lambda v: '__label__{}'.format(v))

    # save train and test set
    dfs = []
    exts = ['.trn', '.tst']
    if settings.VALID_PER_CLASS:
        exts.append('.val')
    for i, ext in enumerate(exts):
        path = titles_out_path + ext
        adf = df.loc[df['test'] == i]
        adf.to_csv(
            path, columns=['label', 'titlen'], index=False, sep=' ',
            header=False, quoting=QUOTE_NONE, escapechar=' '
        )
        dfs.append({
            'path': path,
            'df': adf,
            'ext': ext,
        })

    print(
        ', '.join([
            '{} {}'.format(len(df['df']), df['ext'])
            for df in dfs
        ]),
        '({} test classes)'.format(len(dfs[1]['df']['cat'].value_counts()))
    )

    return dfs

def learn_embeddings_from_transcipts():
    model = fasttext.train_unsupervised(settings.TRANSCRIPTS_PATH)
    print(len(model.words))
    model.save_model(settings.TRANSCRIPTS_MODEL_PATH)

def get_confusion_matrix(preds):

    ret = pd.crosstab(
        preds['cat'],
        preds['pred'],
        rownames=['Actual'],
        colnames=['Predicted'],
        margins=True,
        # won't work, still a bug in pandas
        dropna=False
    )

    # fix for missing columns
    i = 0
    cols = ret.columns.tolist()
    rows = ret.index.tolist()
    labels = sorted(list(set(cols + rows)))
    for i, label in enumerate(labels):
        if label not in cols:
            ret.insert(i, label, 0)
        if label not in rows:
            ret.loc[label] = 0

    ret = ret.sort_index()

    return ret


def get_exp_key():
    auto = ''
    if settings.VALID_PER_CLASS:
        auto = '-{}a'.format(settings.AUTOTUNE_DURATION)

    return '{}l-{}d-{}ep-{}tr-{}{}'.format(
        settings.CAT_DEPTH,
        settings.DIMS,
        settings.EPOCHS,
        settings.TRAIN_REPEAT,
        settings.EMBEDDING_FILE,
        auto
    )


def render_confusion(df_confusion, preds, fmt='g', vmax=None, fname='conf'):
    import seaborn as sn
    import matplotlib.pyplot as plt
    import numpy as np

    # print(df_confusion)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    df_confusion[df_confusion == 0] = np.nan

    if vmax is None:
        vmax = max(df_confusion.iloc[0])*1.5

    sn.heatmap(
        df_confusion,
        annot=True,
        vmax=vmax,
        fmt=fmt,
        ax=ax1,
        annot_kws={'size': 5},
        cmap='Blues',
        linecolor='#ccc',
        linewidths=0.5,
    )
    # plt.show()
    ax1.title.set_text('{}% accuracy'.format(
        int(len(preds.loc[preds['pred'] == preds['cat']]) / len(preds) * 100)
    ))

    ax1.tick_params(axis='both', which='both', labelsize=6)

    import numpy as np
    ax1.set_yticks(np.arange(len(df_confusion)))
    ax1.set_xticks(np.arange(len(df_confusion.columns.tolist())))
    ax1.set_yticklabels(df_confusion.index.tolist())
    ax1.set_xticklabels(df_confusion.columns.tolist(), rotation=90)
    ax1.set_xticks([float(n) + 0.5 for n in ax1.get_xticks()])
    ax1.set_yticks([float(n) + 0.5 for n in ax1.get_yticks()])

    ax1.xaxis.tick_top()

    plt.savefig(get_data_path(settings.PLOT_PATH, get_exp_key() + '-' + fname + '.svg'))

def render_confidence_matrix(preds):

    ret = []

    cats = sorted(list(set(preds['pred']).union(set(preds['cat']))))

    ret.append({
        conf: len(preds[(preds['cat'] == preds['pred']) & (preds['conf'] >= conf)]) / max(len(preds[preds['conf'] >= conf]), 0.1)
        for conf in settings.REPORT_CONFIDENCES
    })
    ret[-1]['cat'] = 'All P'
    ret.append({
        conf: len(preds[(preds['cat'] == preds['pred']) & (preds['conf'] >= conf)]) / len(preds)
        for conf in settings.REPORT_CONFIDENCES
    })
    ret[-1]['cat'] = 'All R'

    for cat in cats:
        cpreds = preds[preds['pred'] == cat]
        ret.append({
            conf: len(cpreds[(cpreds['cat'] == cat) & (cpreds['conf'] >= conf)]) / max(len(cpreds[cpreds['conf'] >= conf]), 0.1)
            for conf in settings.REPORT_CONFIDENCES
        })
        ret[-1]['cat'] = cat + ' P'
        ret.append({
            conf: len(cpreds[(cpreds['cat'] == cat) & (cpreds['conf'] >= conf)]) / max(len(preds[preds['cat'] == cat]), .1)
            for conf in settings.REPORT_CONFIDENCES
        })
        ret[-1]['cat'] = cat + ' R'


    ret = pd.DataFrame(ret).set_index('cat', drop=True)

    render_confusion(ret, preds, '.2f', 1.0, fname='roc')

    return ret
