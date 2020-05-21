import os
from src import utils
from _csv import QUOTE_NONE
import re
import pandas as pd
from src import settings

# load titles.txt file into dataframe
titles_path = utils.get_data_path('in', 'titles.txt')
df = utils.read_df_from_titles(titles_path)

# save that as a csv
titles_out_path = utils.get_data_path('out', 'titles.csv')
df.to_csv(titles_out_path, columns=['id', 'cat1', 'cat2', 'title'], index=False)

df = utils.save_sets(df, 3)

df['titlen'] = df['title'].apply(lambda v: re.sub(r'\W', ' ', v.lower()))

# save training set
df['label'] = df['cat'].apply(lambda v: '__label__{}'.format(v))
train_path = titles_out_path + '.trn'
df_train = df.loc[df['test'] == 0]
df_train.to_csv(train_path, columns=['label', 'titlen'], index=False, sep=' ', header=False, quoting=QUOTE_NONE,
                escapechar=' ')

# save test set
test_path = titles_out_path + '.tst'
df_test = df.loc[df['test'] == 1]
df_test.to_csv(test_path, columns=['label', 'titlen'], index=False, sep=' ', header=False, quoting=QUOTE_NONE,
               escapechar=' ')

print('{} training, {} testing'.format(len(df_train), len(df_test)))

import fasttext

model = fasttext.train_supervised(train_path, epoch=settings.EPOCHS)

# print(model.get_nearest_neighbors('erect'))

# exit()

# print(model.words)
# print(model.labels)

# utils.print_results(*model.test(test_path))

acc = 0
acc_sure = 0
for idx, row in df_test.iloc[0:].iterrows():
    res = model.predict(row['titlen'])
    corr = '<>'
    if row['label'] == res[0][0]:
        acc += 1
        corr = '=='
    print(corr, res, row['label'], row['titlen'])

print(acc / len(df_test))
