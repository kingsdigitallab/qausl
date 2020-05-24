import os
import utils
from _csv import QUOTE_NONE
import re
import pandas as pd
import settings
import fasttext
import time

utils.get_class_titles(1)
# exit()

# utils.extract_transcripts_from_pdfs()
# exit()


# load titles.txt file into dataframe
titles_path = utils.get_data_path('in', 'titles.txt')

if not os.path.exists(titles_path):
    utils.log_error('The training file ({0}) is missing. See README.md for more info.'.format(titles_path))

df = utils.read_df_from_titles(titles_path)

# save that as a csv
titles_out_path = utils.get_data_path('out', 'titles.csv')
df.to_csv(titles_out_path, columns=['id', 'cat1', 'cat2', 'title'], index=False)

# augment dataframe with preprocessed fields
# normalise the title
df['titlen'] = df['title'].apply(lambda v: re.sub(r'\W', ' ', v.lower()))

def train_and_test(df):
    ret = {}

    # randomly split the dataset
    df = utils.split_dataset(
        df,
        settings.CAT_DEPTH,
        settings.TRAIN_PER_CLASS,
        settings.TEST_PER_CLASS,
        settings.VALID_PER_CLASS,
    )

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

    options = {
    }

    if settings.VALID_PER_CLASS:
        options['autotuneValidationFile'] = dfs[2]['path']
        options['autotuneDuration'] = settings.AUTOTUNE_DURATION
    else:
        options['epoch'] = settings.EPOCHS
        options['dim'] = settings.DIMS

    if settings.EMBEDDING_FILE:
        options['pretrainedVectors'] = utils.get_data_path('in', settings.EMBEDDING_FILE)

    model = fasttext.train_supervised(
        dfs[0]['path'],
        **options
    )

    acc = 0
    sure = 0
    sure_correct = 0
    for idx, row in dfs[1]['df'].iloc[0:].iterrows():
        res = model.predict(row['titlen'])
        corr = '<>'
        confidence = res[1][0]
        if confidence > settings.MIN_CONFIDENCE:
            sure += 1
        if row['label'] == res[0][0]:
            acc += 1
            corr = '=='
            if confidence > settings.MIN_CONFIDENCE:
                sure_correct += 1
        elif confidence > settings.MIN_CONFIDENCE:
            corr = '!!'
        # print(corr, res, row['label'], row['titlen'])

    if sure < 1:
        sure = 0.001

    print('acc: {:.2f} certain: {:.2f} acc certain: {:.2f} {:.2f}'.format(
        acc / len(dfs[1]['df']),
        sure / len(dfs[1]['df']),
        sure_correct / sure,
        sure_correct / len(dfs[1]['df'])
    ))

    ret['acc'] = acc / len(dfs[1]['df'])

    return ret

ress = []
t0 = time.time()
for i in range(0, settings.TRAIN_REPEAT):
    print('trial {}/{}'.format(i + 1, settings.TRAIN_REPEAT))
    ress.append(train_and_test(df))
t1 = time.time()

accs = [r['acc'] for r in ress]
acc_avg = sum(accs) / len(ress)
acc_min = min(accs)
acc_max = max(accs)

print('avg: {:.2f} [{:.2f}, {:.2f}], depth: {}, {} trials, {} dims, {} epochs, (Embedddings: {}), {:.1f} minutes.'.format(
    acc_avg, acc_min, acc_max,
    settings.CAT_DEPTH,
    settings.TRAIN_REPEAT,
    settings.DIMS, settings.EPOCHS, settings.EMBEDDING_FILE,
    (t1-t0)/60
))
