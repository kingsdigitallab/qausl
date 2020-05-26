import os
import utils
import re
import pandas as pd
import settings
import fasttext
import time

if 0:
    utils.extract_transcripts_from_pdfs()
    exit()

if 0:
    utils.learn_embeddings_from_transcipts()
    exit()


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

def train_and_test(df, preds):
    ret = {}

    # randomly split the dataset
    df = utils.split_dataset(
        df,
        settings.CAT_DEPTH,
        settings.TRAIN_PER_CLASS,
        settings.TEST_PER_CLASS,
        settings.VALID_PER_CLASS,
    )

    # prep sets for FT and save them on disk
    dfs = utils.save_ft_sets(df, titles_out_path)

    options = {}

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

        def short_label(full_label):
            return full_label.split('__')[-1]

        preds.append({
            'cat': short_label(row['label']),
            'pred': short_label(res[0][0]),
            'conf': res[1][0],
            'title': row['titlen'],
        })

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
preds = []
for i in range(0, settings.TRAIN_REPEAT):
    print('trial {}/{}'.format(i + 1, settings.TRAIN_REPEAT))
    ress.append(train_and_test(df, preds))
t1 = time.time()

preds = pd.DataFrame(preds)

if 1:
    df_confusion = utils.get_confusion_matrix(preds)
    utils.render_confusion(df_confusion, preds)

if 1:
    utils.render_confidence_matrix(preds)

if 1:
    acc = len(preds.loc[preds['pred'] == preds['cat']]) / len(preds)
    utils.log('{}, {:.2f} acc, {:.1f} minutes.'.format(
        utils.get_exp_key(),
        acc,
        (t1-t0)/60
    ))
