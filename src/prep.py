import os
from _csv import QUOTE_MINIMAL, QUOTE_NONE

path = os.path.dirname(__file__)
data_path = os.path.join(path, '..', 'data')
titles_name = 'titles.txt'
titles_path = os.path.join(data_path, 'in', titles_name)

with open(titles_path, 'rt') as fh:
    content = fh.read()

import re
titles = re.findall(r'(?m)^\s*(.*?)\s*(\d+)\s*\(([^)]+)\)[^(]*$', content)

diff = set(range(1, 455)) - set([int(t[1]) for t in titles])
if diff:
    print(len(titles))
    print(titles[0:2])
    print(sorted(diff))
    exit()

titles_out_path = os.path.join(data_path, 'out', 'titles.csv')
import pandas as pd
labels = ['title', 'id', 'cat']
df = pd.DataFrame.from_records(titles, columns=labels)

# df['cat1'], df['cat2'] = \
df = df.join( df['cat'].str.split(r'/', 1, expand=True)).rename(columns={0:'cat1', 1:'cat2'})

df['cat1'].replace({'73': '703'}, inplace=True)

def save_sets(df, depth=3, cat_test=2):
    # shuffle the data
    df = df.sample(frac=1, random_state=None).reset_index(drop=True)
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
    train = pd.DataFrame()
    for idx, row in df.iterrows():
        cat = row['cat']
        left = vc.get(cat, 0)
        if left:
            vc[cat] = left - 1
            df.loc[idx, 'test'] = 1

    # print(len(df[df['test'] == 1]))

    return df

# print(df[0:1])

# save while set
df.to_csv(titles_out_path, columns=['id', 'cat1', 'cat2', 'title'], index=False)

df = save_sets(df, 3)

import re
df['titlen'] = df['title'].apply(lambda v: re.sub(r'\W', ' ', v.lower()))

# save training set
df['label'] = df['cat'].apply(lambda v: '__label__{}'.format(v))
train_path = titles_out_path+'.trn'
df_train = df.loc[df['test'] == 0]
df_train.to_csv(train_path, columns=['label', 'titlen'], index=False, sep=' ', header=False, quoting=QUOTE_NONE, escapechar=' ')

# save test set
test_path = titles_out_path+'.tst'
df_test = df.loc[df['test'] == 1]
df_test.to_csv(test_path, columns=['label', 'titlen'], index=False, sep=' ', header=False, quoting=QUOTE_NONE, escapechar=' ')

print('{} training, {} testing'.format(len(df_train), len(df_test)))

import fasttext

model = fasttext.train_supervised(train_path, epoch=500)

print(model.get_nearest_neighbors('erect'))

# exit()

print(model.words)
print(model.labels)

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

print_results(*model.test(test_path))

acc = 0
acc_sure = 0
for idx, row in df_test.iloc[0:].iterrows():
    res = model.predict(row['titlen'])
    corr = '<>'
    if row['label'] == res[0][0]:
        acc += 1
        corr = '=='
    print(corr, res, row['label'], row['titlen'])

print(acc/len(df_test))


