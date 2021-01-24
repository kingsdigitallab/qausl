import math
import os
import re
import urllib

import pandas as pd
from matplotlib import ticker

import settings
import seaborn as sn
from _csv import QUOTE_NONE
import nltk
import matplotlib.pyplot as plt
import numpy as np


STEMMER = nltk.stem.porter.PorterStemmer()
LEMMATIZER = nltk.stem.wordnet.WordNetLemmatizer()
STOP_WORDS = nltk.corpus.stopwords.words("english")
STOP_WORDS.extend(
            'further chap sic act supplement resolution entituled entitled chapter section'.split())

class Bidict(dict):
    '''A python dict with a method to get a key from a value'''

    def get_key_from_val(self, val, default=None):
        ret = default
        keys = [k for k, v in self.items() if v == val]
        if keys:
            ret = keys[0]
        return ret


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
        with open(path, 'rt', encoding='utf-8', errors='replace') as fh:
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
        title, id, cat1, cat2
    cat1 is the primary 3 digit category, e.g. 156
    cat2 is an optional secondary category for the same chapter. NaN if none.
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

    # IMPORTANT: left pad cat1 & cat2 with 0s
    def clean_cat(cat):
        # e.g. 0.2 => 002
        return re.sub(r'\..*', r'', str(cat))[:3].zfill(3)
    df['cat1'] = df['cat1'].apply(clean_cat)
    df['cat2'] = df['cat2'].apply(clean_cat)

    return df


def read_df_from_titles_txt(path, use_full_text):
    '''returns a pandas dataframe from titles.txt

    e.g. of an input line:

    An ACT for establishing a nightly watch [...] Philadelphia. 21 (601/603)

    dataframe columns:
        title, id, cat1, cat2
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

    labels = ['title', 'id', 'cat_slash']
    df = pd.DataFrame.from_records(titles, columns=labels)

    # split multiple categories into cat1 and cat2
    df = df.join(df['cat_slash'].str.split(r'/', 1, expand=True)).rename(columns={0: 'cat1', 1: 'cat2'})

    # bug in the input file: 73 stands for 703, it seems
    # df['cat1'].replace({'73': '703'}, inplace=True)

    return df


def split_dataset(df, depth=3, cat_train=2, cat_test=2, cat_valid=0, seed=None):
    '''
    Shuffles and split the dataset into training and testing samples.
    Uses a new column 'split' =
        -1: sample unused
        0: training sample
        1: testing sample
        2: validating sample

    :param df: dataframe with all titles
    :param depth: depth of the taxonomy for the classification
    :param cat_test: minimum number of test sample per class
    :return: shuffled dataframe with a new column 'split': -1,0,1 or 2

    cat1 & cat2 will be strings with <depth> digits.
    '''
    # Shuffle the data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Make sure the categories have the expected format
    df['cat1'] = df['cat1'].apply(lambda v: str(v)[:depth])
    df['cat2'] = df['cat2'].apply(lambda v: str(v)[:depth])

    # Count samples per cat
    vc = df['cat1'].value_counts()
    render_category_distribution(vc)

    # only test or validate on cats which have enough samples
    # vc = vc.where(lambda x: x >= (cat_train + cat_test + cat_valid)).dropna()

    vc = {k: [settings.TRAIN_PER_CLASS_MAX, cat_test, cat_valid] for k in vc.index}

    train_threshold = (settings.TRAIN_PER_CLASS_MAX - settings.TRAIN_PER_CLASS_MIN)

    # Actual splitting.
    # -1:unassigned; 0:train; 1:test; 2:valid
    # TODO: support for valid
    df['split'] = -1
    for idx, row in df.iterrows():
        cat = row['cat1']
        split = -1
        if row['can_test'] and vc[cat][1]:
            # collect test samples first
            split = 1
        elif row['can_train'] and vc[cat][2] and vc[cat][0] <= train_threshold:
            # collect valid samples if min training collected
            split = 2
        elif row['can_train'] and vc[cat][0]:
            # collect min to max training samples
            split = 0
        if split > -1:
            vc[cat][split] -= 1
            df.loc[idx, 'split'] = split

    # don't test categories without enough test & train samples
    tiny_cats = []
    for cat, counts in vc.items():
        if counts[1] or counts[0] > train_threshold:
            df.loc[(df.cat1 == cat) & (df.split == 1), 'split'] = -1
            tiny_cats.append(cat)

    if tiny_cats:
        print(f'Untestable cats: {tiny_cats}')

    if settings.GROUP_TINY_CLASSES:
        group_cat = '#' * depth
        df.loc[df.cat1.isin(tiny_cats), 'cat1'] = group_cat
        df.loc[df.cat2.isin(tiny_cats), 'cat2'] = group_cat
        # split this group into testing and training
        indexes = df.loc[df.cat1 == group_cat].index
        df.loc[df.index.isin(indexes[0:cat_test]), 'split'] = 1
        df.loc[df.index.isin(indexes[cat_test:cat_test+settings.TRAIN_PER_CLASS_MAX]), 'split'] = 0

    # print(sorted(tiny_cats2))

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
    # remove small words
    ret = re.sub(r'\b\w{1,2}\b', r' ', ret)
    # remove digits
    # TODO: use D
    ret = re.sub(r'\d+', r' ', ret)
    # remove non-discriminant words
    ret = re.sub(r'\b(further|chap|sic|a|of|and|an|the|to|act|supplement|for|resolution|entituled|chapter|section)\b', '', ret)
    # remove all non-words
    ret = re.sub(r'\W+', ' ', ret)
    return ret


def read_class_titles(depth):
    '''Returns a dictionary of categories at given depth.
    {cat: title}

    cat is a string with depth digits legal category code.
    '''
    content = read_file(get_data_path('in', 'classes.txt'))

    return Bidict({
        cls[0]: re.sub('\W+', ' ', cls[1])
        for cls in re.findall(r'(?m)^(\d+)\.?\s+(.+)$', content)
        if len(cls[0]) == depth
    })


def save_ft_sets(df, titles_out_path):
    # prepare the label for fasttext format
    df['label'] = df['cat1'].apply(lambda v: '__label__{}'.format(v))

    # save train and test set
    dfs = []
    exts = ['.trn', '.tst']
    # if settings.VALID_PER_CLASS:
    exts.append('.val')
    for i, ext in enumerate(exts):
        path = titles_out_path + ext
        adf = df.loc[df['split'] == i]
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
        ' | '.join([
            '{} {} ({} cats)'.format(
                len(df['df']),
                df['ext'],
                df['df'].cat1.nunique(),
            )
            for df in dfs
            if len(df['df'])
        ])
    )

    return dfs


def learn_embeddings_from_transcipts():
    import fasttext
    model = fasttext.train_unsupervised(settings.TRANSCRIPTS_PATH)
    print(len(model.words))
    model.save_model(settings.TRANSCRIPTS_MODEL_PATH)


def get_confusion_matrix(preds, classifier):

    ret = pd.crosstab(
        preds['cat'],
        preds['pred'],
        rownames=['Actual'],
        colnames=['Predicted'],
        # margins=True,
        # won't work, still a bug in pandas
        dropna=False
    )

    # fix for missing columns
    cols = ret.columns.tolist()
    rows = ret.index.tolist()
    labels = sorted(list(set(cols + rows)))
    for i, label in enumerate(labels):
        if label not in cols:
            ret.insert(i, label, 0)
        if label not in rows:
            ret.loc[label] = 0

    ret = ret.sort_index()

    # add a row with number of False Positives
    import numpy as np
    ret.loc['FP'] = ret.sum() - np.diag(ret)

    # add a row with number of training samples
    ret.loc['Trained'] = classifier.df_train.cat1.value_counts()

    return ret


def get_exp_key(classifier):
    '''Returns a string that summarises the settings of the
    training process.'''
    auto = ''
    if settings.VALID_PER_CLASS and settings.CLASSIFIER == 'FastText':
        auto = settings.AUTOTUNE_DURATION

    filenames = '-'.join([
        (
            ('trn_' if info['can_train'] else '') +
            ('tst_' if info['can_test'] else '') +
            info['filename'].replace('titles-', '')
        )
        for info
        in settings.DATASET_FILES
        if info['can_train'] or info['can_test']
    ])

    parts = [
        ['', classifier.__class__.__name__],
        ['l', settings.CAT_DEPTH],
        ['ep', settings.EPOCHS],
        ['tr', settings.TRIALS],
        ['trn', f'{settings.TRAIN_PER_CLASS_MIN}-{settings.TRAIN_PER_CLASS_MAX}'],
        ['tst', settings.TEST_PER_CLASS],
        ['val', settings.VALID_PER_CLASS],
        ['', filenames],
        ['', 'fulltext' if settings.FULL_TEXT else ''],
        ['dim', classifier.get_internal_dimension()],
        ['', classifier.get_pretrained_model_name()],
        ['auto', auto],
        ['class_weight', settings.CLASS_WEIGHT]
    ]

    ret = '_'.join([
        f'{val}{suffix}'
        for suffix, val in
        parts
        if val
    ])

    return ret


def render_confusion(classifier, df_confusion, preds, fmt='g', vmax=None, fname='conf'):
    import matplotlib.pyplot as plt
    import numpy as np

    # print(df_confusion)

    number_of_classes = len(df_confusion)
    fig = plt.figure(figsize=[s / 25 * number_of_classes for s in [15, 10]])
    # ax1 = fig.add_subplot(111)

    # trick not to show the '0 annotation in those 0 cells
    df_confusion[df_confusion == 0] = np.nan

    tests_per_class = settings.TEST_PER_CLASS * settings.TRIALS

    if vmax is None:
        vmax = tests_per_class

    ax1 = sn.heatmap(
        df_confusion,
        annot=True,
        vmax=vmax,
        vmin=0.0,
        fmt=fmt,
        # ax=ax1,
        annot_kws={'size': 8},
        cmap='Blues',
        linecolor='#ccc',
        linewidths=0.5,
        cbar=False,
    )
    # plt.show()
    ax1.title.set_text('Confusion matrix ({}% accuracy) - {} tests/cat.'.format(
        int(len(preds.loc[preds['pred'] == preds['cat']]) / len(preds) * 100),
        tests_per_class,
    ))
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=30)

    plt.savefig(get_data_path(settings.PLOT_PATH, get_exp_key(classifier) + '-' + fname + '.svg'))
    plt.close()


def render_confusion_old(df_confusion, preds, fmt='g', vmax=None, fname='conf'):
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


def render_confidence_matrix(classifier, preds):
    '''
    Render a precision vs recall chart for each category.

    :param preds: a dataframe with predictions, columns:
        cat: true class
        pred: predicted class
        conf: level of confidence of the prediction
    '''

    ret = []

    # preds -> precs[recall, cat] = precision
    precs = pd.DataFrame()

    cats = sorted(preds.cat.unique())
    for cat in ['AVG'] + cats:
        for conf in settings.REPORT_CONFIDENCES:
            relevant = preds
            sure = preds.loc[preds.conf >= conf]
            if cat != 'AVG':
                relevant = relevant.loc[preds.cat == cat]
                sure = sure.loc[preds.cat == cat]
            sure_correct = sure.loc[preds.pred == preds.cat]

            recall = len(sure_correct) / len(relevant)
            try:
                precision = len(sure_correct) / len(sure)
            except ZeroDivisionError:
                precision = math.nan
            precs.loc[recall, cat] = precision

    precs.loc[0] = 1.0

    precs = precs.sort_index()
    precs = precs.interpolate()

    # Remove all the categories which have lines going below the average (AVG)
    # at the median point or at the bottom point in confidence/precision.
    # This makes the chart easier to read at L£ with so many categories.
    hide_cats_above_ALL = len(cats) > 10
    if hide_cats_above_ALL:
        # median confidence point
        meds = precs.median()
        meds_cats = meds.loc[meds < meds[0]].index
        # lowest confidence point
        bots = precs.iloc[-1]
        bots_cats = bots.loc[bots < bots[0]].index
        cats_below_ALL = ['AVG']
        # union (i.e. OR)
        cats_below_ALL.extend(meds_cats.union(bots_cats).sort_values().to_list())
        precs = precs[cats_below_ALL]

    fig = plt.figure(figsize=[10, 10])
    sn.set_style("ticks")
    ax1 = sn.lineplot(data=precs)

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.set(xlabel='Recall', ylabel='Precision')

    # show category on teh right of each line
    for cat in precs.columns:
        ax1.annotate(
            str(cat),
            # (p.get_width() + 3, p.get_y() + 0.5),
            (1.0, precs.iloc[-1][cat]),
            ha='left', va='bottom',
            color='black'
        )

    ax1.title.set_text('Precision / recall ({}, {}% accuracy)'.format(
        classifier.__class__.__name__,
        int(len(preds.loc[preds['pred'] == preds['cat']]) / len(preds) * 100),
    ))

    fname = 'prec'

    plt.savefig(get_data_path(settings.PLOT_PATH, get_exp_key(classifier) + '-' + fname + '.svg'))
    plt.close()

    return ret


def render_confidence_matrix_old(classifier, preds):
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

    render_confusion(classifier, ret, preds, '.2f', 1.0, fname='roc')

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


def normalise_roman_number(roman_number):
    '''returns a normalised string, which is more lenient for frequent
    OCR errors. This normalisation will help comparing roman numbers
    without much risk of false positives.'''
    # replace nonsensical I/L, e.g. 1492: MCDLXCII => MCDXCII
    return roman_number.upper().replace('O', 'C').replace('T', 'I').replace('L', 'I').replace('Y', 'V').replace('1', 'I').replace('IXC', 'XC')


def render_category_distribution(categories_count):
    '''
    :param categories_count: A panda Series: category -> number of titles
    Save the distribution as a SVG.
    '''
    categories_count = categories_count.copy()
    cat_num = len(categories_count)

    tiny_threshold = settings.TRAIN_PER_CLASS_MIN + settings.TEST_PER_CLASS + settings.VALID_PER_CLASS
    tiny_categories = categories_count[categories_count < tiny_threshold]
    if settings.GROUP_TINY_CLASSES:
        categories_count['#' * settings.CAT_DEPTH] = tiny_categories.sum()
    to_num = f' -> {cat_num - len(tiny_categories)}'

    depth = len(categories_count.first_valid_index())
    # save as bar chart
    vc_df = categories_count.to_frame().reset_index()
    vc_df = vc_df.rename(columns={'cat1': 'chapters'})
    vc_df['category'] = vc_df['index']
    sn.set_theme(style='whitegrid')

    from matplotlib import pyplot
    # make graph taller so bars and their labels are readable
    pyplot.figure(figsize=(10, 20/3*depth))

    options = {}
    if depth > 1:
        vc_df['Level 1 category'] = vc_df['index'].apply(lambda x: x[0])
        options['hue'] = 'Level 1 category'

    ax = sn.barplot(
        y='category', x='chapters',
        data=vc_df,
        dodge=False, **options
    )

    ax.set_title(f'Number of chapters per level {depth} category. ({cat_num}{to_num} categories)')

    # show count next to each bar
    for p in ax.patches:
        if not math.isnan(p.get_width()):
            ax.annotate(
                str(int(p.get_width())),
                (p.get_width() + 3, p.get_y() + 0.5),
                ha='left', va='bottom',
                color='black'
            )

    # draw threshold for tiny categories
    if tiny_threshold:
        ax.axvline(tiny_threshold)

    ax.axvline(settings.TRAIN_PER_CLASS_MAX + settings.TEST_PER_CLASS)

    fig = ax.get_figure()
    fig.savefig(get_data_path('out', f'cat-{depth}.svg'))
    pyplot.close()

    # save as CSV
    vc_df.to_csv(
        get_data_path('out', f'cat-{depth}.csv'),
        columns=['category', 'chapters'],
        index=False,
    )


def short_label(full_label):
    '''Returns XXX from label__XXX.
    label__XXX is fasttext format for training sample labels.'''
    return full_label.split('__')[-1]


def repair_ocred_text(text):
    '''try to mend text OCRed from a PDF.'''
    ret = text

    # reunite hyphenated words (de-Syllabification)
    ret = re.sub(r'(\w)\s*-\s*(\w)', r'\1\2', ret)

    # quotation marks
    ret = ret.replace('”', '"').replace('“', '"')
    
    # 752 The Statutes at Large of Pennsylvania. [1808
    # 845 846 The Statutes at Large of Pennsylvania. [1808
    pattern = r'[(){}\s\d\[\]\.]+The Statutes at Large of Pennsylvania[(){}\s\d\[\]\.]+'
    # print(re.findall(pattern, ret))
    ret = re.sub(pattern, ' ', ret)

    # remove small bracketed content
    # e.g. {Section I.] (Section II, P. L.)
    pattern = r'[\[{(][^)}\]]{1,20}[)\]}]'
    # ms = re.findall(pattern, ret)
    # if ms:
    #     print('bracketed', ms)
    ret = re.sub(pattern, ' ', ret)

    # remove line breaks
    ret = re.sub(r'\s+', r' ', ret)

    return ret.strip()


def tokenise(
    string,
    stemmatise = False,
    lemmatise = False,
    stop_words = False,
    no_small_words = False,
    no_numbers = False
):
    '''Simplify the input string to help improve training.'''

    # clean (convert to lowercase and remove punctuations
    # and characters and then strip)
    ret = re.sub(r'[^\w\s]', '', str(string).lower().strip())

    if no_numbers:
        ret = re.sub(r'\d+', ' ', ret)

    if no_small_words:
        ret = re.sub(r'\b\w{1,2}\b', ' ', ret)

    ## Tokenize (convert from string to list)
    ret = ret.split()
    ## remove Stopwords
    if stop_words:
        ret = [
            word for word in ret
            if word not in STOP_WORDS
        ]

    ## Stemming (remove -ing, -ly, ...)
    if stemmatise:
        ret = [STEMMER.stem(word) for word in ret]

    ## Lemmatisation (convert the word into root word)
    if lemmatise:
        ret = [LEMMATIZER.lemmatize(word) for word in ret]

    ## back to string from list
    return " ".join(ret)


def fix_torch_cuda_long_wait():
    '''On my laptop, when the eGPU is off, some pytorch imports
    will hang for a very long time in this function:

    torch.cuda.is_available()

    Here we monkey patch torch to avoid that. Other solutions proposed on
    stackoverflow didn't work. https://stackoverflow.com/q/53266350
    '''
    import torch
    if settings.CPU_ONLY:
        delattr(torch._C, '_cuda_getDeviceCount')


def download(url, out_path):
    '''Download the resource at url into a file at out_path.
    If file exists, do nothing.
    Returns 1 if file already exists. 2 if downloaded. 0 on error.'''
    ret = 1

    if not os.path.exists(out_path):
        ret = 0
        try:
            with urllib.request.urlopen(url) as resp:
                with open(out_path, 'wb') as fh:
                    fh.write(resp.read())
                ret = 2
        except urllib.error.HTTPError as e:
            print(f"ERROR: {url} {e}")

    return ret


def set_global_seed(seed=None):
    if seed:
        import random
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        import tensorflow as tf
        tf.random.set_seed(seed)
