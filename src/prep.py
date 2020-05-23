import os
from src import utils
from _csv import QUOTE_NONE
import re
import pandas as pd
from src import settings
import fasttext

# load titles.txt file into dataframe
titles_path = utils.get_data_path('in', 'titles.txt')
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
    df = utils.split_dataset(df, settings.CAT_DEPTH)

    # prepare the label for fasttext format
    df['label'] = df['cat'].apply(lambda v: '__label__{}'.format(v))

    # save train and test set
    dfs = []
    for i in [0, 1]:
        path = titles_out_path + ('.tst' if i else '.trn')
        adf = df.loc[df['test'] == i]
        adf.to_csv(
            path, columns=['label', 'titlen'], index=False, sep=' ',
            header=False, quoting=QUOTE_NONE, escapechar=' '
        )
        dfs.append({
            'path': path,
            'df': adf,
        })

    print('{} training, {} testing'.format(len(dfs[0]['df']), len(dfs[1]['df'])))

    options = {
        'dim': settings.DIMS,
    }

    if settings.EMBEDDING_FILE:
        options['pretrainedVectors'] = utils.get_data_path('in', settings.EMBEDDING_FILE)

    model = fasttext.train_supervised(dfs[0]['path'], epoch=settings.EPOCHS, **options)

    # print(model.get_nearest_neighbors('erect'))
    # print(model.words)
    # print(model.labels)
    # utils.print_results(*model.test(test_path))

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

    print('acc: {:.2f} certain: {:.2f} acc certain: {:.2f} {:.2f}'.format(
        acc / len(dfs[1]['df']),
        sure / len(dfs[1]['df']),
        sure_correct / sure,
        sure_correct / len(dfs[1]['df'])
    ))

    ret['acc'] = acc / len(dfs[1]['df'])

    return ret

ress = []
for i in range(0, settings.TRAIN_REPEAT):
    ress.append(train_and_test(df))

accs = [r['acc'] for r in ress]
acc_avg = sum(accs) / len(ress)
acc_min = min(accs)
acc_max = max(accs)

print('avg: {:.2f} [{:.2f}, {:.2f}]'.format(acc_avg, acc_min, acc_max))
