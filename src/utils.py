import os
import re
import pandas as pd
import settings
import fasttext
import seaborn as sn
from _csv import QUOTE_NONE

def get_data_path(*path_parts):
    '''e.g. get_data_path('in', 'myfile.txt', makedirs=True)
    Will return the absolute path to data/in/myfile.txt
    and create the intermediate folders if makedirs is True
    '''

    ret = os.path.join(settings.DATA_PATH, *path_parts)

    return ret


def read_file(path):
    ret = ''

    if os.path.exists(path):
        with open(path, 'rt', encoding='utf-8') as fh:
            ret = fh.read()

    return ret


def read_df_from_titles(path, use_full_text):
    fct = None
    if path.endswith('.txt'):
        fct = read_df_from_titles_txt
    if path.endswith('.csv'):
        fct = read_df_from_titles_csv

    return fct(path, use_full_text)


def read_df_from_titles_csv(path, use_full_text):
    '''returns a pandas dataframe from titles.csv
    dataframe columns:
        title, id, cat, cat1, cat2
    cat1 is the 3 digit category, e.g. 156
    cat2 is an alternative categorisation for the same chapter.
    '''
    df = pd.read_csv(path)

    input_column = 'Title'
    if use_full_text:
        input_column = 'Text'

    # split multiple categories into cat1 and cat2
    df = df.rename(columns={
        'Chapter': 'id',
        'Class 3': 'cat1',
        '1st alt Class 3': 'cat2',
        input_column: 'title',
    })

    # IMPORTANT: pad cat1 & cat2 with 0s
    df['cat1'] = df['cat1'].apply(lambda c: str(c).zfill(3))
    df['cat2'] = df['cat2'].apply(lambda c: str(c).zfill(3))

    return df


def read_df_from_titles_txt(path, use_full_text):
    '''returns a pandas dataframe from titles.txt

    e.g. of an input line:

    An ACT for establishing a nightly watch [...] Philadelphia. 21 (601/603)

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
    # df['cat1'].replace({'73': '703'}, inplace=True)

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
    df['cat'] = df['cat1'].apply(lambda v: str(v)[:depth])
    # count cats
    vc = df['cat'].value_counts()

    save_category_distribution(vc)

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


def extract_text_from_pdf(pdf_path, use_tesseract=False):
    '''
    :param pdf_path: path to a pdf file
    :param use_tesseract: if True, OCR with tesseract;
        extract embedded text otherwise.
    :return: utf-8 string with the text content of the pdf
    '''
    if use_tesseract:
        import textract
        try:
            ret = textract.process(
                pdf_path,
                method='tesseract',
                language='eng',
                encoding='utf-8'
            )
            ret = ret.decode('utf8')
        except UnicodeDecodeError as e:
            ret = f'ERROR: {e}'
    else:
        import fitz  # this is pymupdf
        with fitz.open(pdf_path) as doc:
            ret = ''
            for page in doc:
                ret += page.getText()

    return ret


def extract_transcripts_from_pdfs():
    fh = open(settings.TRANSCRIPTS_PATH, 'wb')

    from pathlib import Path
    paths = list(Path(get_data_path('in', 'pdfs')).rglob("*TEXT*.pdf"))

    paths = [str(p) for p in paths]
    for p in sorted(paths, key=lambda p: os.path.basename(p)):
        print(p)
        fh.write(('\n\nNEWFILE {}\n\n'.format(os.path.basename(p))).encode('utf-8'))

        content = extract_text_from_pdf(p, use_tesseract=True)
        # content = textract.process(p, method='tesseract', language='eng')

        fh.write(content)

    fh.close()


def tokenise_title(title):
    ret = title.lower()
    # reunite hyphenated words
    ret = re.sub(r'(\w)\s*-\s*(\w)', r'\1\2', ret)
    ret = re.sub(r'\b\w{1,2}\b', r' ', ret)
    ret = re.sub(r'\d+', r' ', ret)
    ret = re.sub(r'\b(further|chap|sic|a|of|and|an|the|to|act|supplement|for|resolution|entituled|chapter|section)\b', '', ret)
    ret = re.sub(r'\W+', ' ', ret)
    return ret


def get_class_titles(depth=3):
    classes = {}

    content = read_file(get_data_path('in', 'classes.txt'))

    for cls in re.findall(r'(?m)^(\d+)\.?\s+(.+)$', content):
        cls_num = cls[0]
        cls = {
            'title': cls[1],
            'titlen': tokenise_title(cls[1]),
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
    # if settings.VALID_PER_CLASS:
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
    '''Returns a string that summarises the settings of the
    training process.'''
    auto = ''
    if settings.VALID_PER_CLASS:
        auto = '-{}a'.format(settings.AUTOTUNE_DURATION)

    return '{}-{}l-{}d-{}ep-{}tr-{}cap-{}ds-{}-{}{}'.format(
        settings.CLASSIFIER,
        settings.CAT_DEPTH,
        settings.DIMS,
        settings.EPOCHS,
        settings.TRIALS,
        settings.TRAIN_PER_CLASS_MAX,
        re.sub(r'^\D+(\d+).*?$', r'\1', settings.TITLES_FILENAME),
        'fulltxt' if settings.FULL_TEXT else 'titles',
        settings.EMBEDDING_FILE,
        auto
    )


def render_confusion(df_confusion, preds, fmt='g', vmax=None, fname='conf'):
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
    '''
    :param preds: a dataframe with predictions, columns:
        cat: true class
        pred: predicted class
        conf: level of confidence of the prediction
    :return: array of recall and precision
        for a range of confidence level
        for all the classes & overall
        [
            {'cat': '1 P', '0': '0.43', '0.5': '0.53'}
            {'cat': '1 R', '0': '0.70', '0.5': '0.67'}
            2 P
            2 R
            ...
        ]
    '''

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


def get_roman_from_int(num):
    num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'),
               (90, 'XC'),
               (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'),
               (4, 'IV'), (1, 'I')]

    ret = ''

    while num > 0:
        for i, r in num_map:
            while num >= i:
                ret += r
                num -= i

    return ret


def save_category_distribution(categories_count):
    '''
    :param categories_count: A panda Series: category -> number of titles
    Save the distribution as a SVG.
    '''
    depth = len(categories_count.first_valid_index())
    # save as bar chart
    vc_df = categories_count.to_frame().reset_index()
    vc_df = vc_df.rename(columns={'cat': 'titles'})
    vc_df['cat1'] = vc_df['index'].apply(lambda x: x[0])
    sn.set_theme(style='whitegrid')

    from matplotlib import pyplot
    # make graph taller so bars and their labels are readable
    pyplot.figure(figsize=(10, 20/3*depth))

    ax = sn.barplot(
        y='index', x='titles',
        data=vc_df.reset_index(),
        hue='cat1', dodge=False,
    )
    fig = ax.get_figure()
    fig.savefig(get_data_path('out', f'cat-{depth}.svg'))

    # save as CSV
    vc_df['category'] = vc_df['index']
    vc_df.to_csv(
        get_data_path('out', f'cat-{depth}.csv'),
        columns=['category', 'titles'],
        index=False,
    )


def short_label(full_label):
    '''Returns XXX from label__XXX.
    label__XXX is fasttext format for training sample labels.'''
    return full_label.split('__')[-1]
