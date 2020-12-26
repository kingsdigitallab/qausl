import os
import utils
import pandas as pd
import settings
import time
from classifiers import FastText, Classifier

titles_out_path = utils.get_data_path('out', 'titles.csv')


def train_and_test(df, preds):
    '''
    Run a single trial:
        Shuffle df and split it into training and testing subsets
        Train a new model based on the training sets
        Test the model with testing set
        Add prediction data into preds array

    :param df: dataframe with full set of all available samples
        columns: id, cat1 (primary class), cat2 (secondary),
        title, titlen (claened title)
    :param preds: an array of predictions, each prediction is a dictionary
        cat: true category, pred: predicted category,
        conf: model confidence in its prediction (< 1.0),
        title: actual title of the chapter/sample
    :return: average testing accuracy
    '''
    ret = {}

    # PREPS
    # randomly split the dataset
    df = utils.split_dataset(
        df,
        settings.CAT_DEPTH,
        settings.TRAIN_PER_CLASS,
        settings.TEST_PER_CLASS,
        settings.VALID_PER_CLASS,
    )

    # TRAIN
    classifier = Classifier.from_name(settings.CLASSIFIER)
    classifier.set_datasets(df, titles_out_path)
    classifier.train(df)

    df_test = classifier.df_test

    # TEST
    acc = 0
    sure = 0
    sure_correct = 0
    for idx, row in df_test.iloc[0:].iterrows():
        pred, confidence = classifier.predict(row['titlen'])

        preds.append({
            'cat': utils.short_label(row['label']),
            'pred': utils.short_label(pred),
            'conf': confidence,
            'title': row['titlen'],
        })

        corr = '<>'
        if confidence > settings.MIN_CONFIDENCE:
            sure += 1
        if row['label'] == pred:
            acc += 1
            corr = '=='
            if confidence > settings.MIN_CONFIDENCE:
                sure_correct += 1
        elif confidence > settings.MIN_CONFIDENCE:
            corr = '!!'
        if corr != '==':
            print('{} actual: {}, pred.: {} ({:.2f} c.) title: {}'.format(
                corr,
                utils.short_label(row['label']),
                utils.short_label(pred),
                confidence,
                row['title'][:100].replace('\n', '')
            ))

    if sure < 1:
        sure = 0.001

    print('acc: {:.2f} certain: {:.2f} acc certain: {:.2f} {:.2f}; {}d'.format(
        acc / len(df_test),
        sure / len(df_test),
        sure_correct / sure,
        sure_correct / len(df_test),
        classifier.get_internal_dimension(),
    ))

    ret['acc'] = acc / len(df_test)

    return ret


def run_trials():
    '''Run multiple trials to obtain more statistically relevant results
    and make best use of our small dataset (using cross-validation).
    For each trial, train a new classifier on a random training sample.
    '''

    df = prepare_dataset()

    t0 = time.time()
    preds = []
    for i in range(0, settings.TRIALS):
        print('trial {}/{}'.format(i + 1, settings.TRIALS))
        train_and_test(df, preds)
        print('-' * 40)
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


def prepare_dataset():
    '''Convert input .txt o .csv into a .csv file with all the necessary
    columns for training and testing classification models.'''

    # # experimental work done on first, small dataset.
    # utils.extract_transcripts_from_pdfs()
    # utils.learn_embeddings_from_transcipts()

    # load titles file into dataframe
    titles_path = utils.get_data_path('in', settings.TITLES_FILENAME)

    if not os.path.exists(titles_path):
        utils.log_error('The training file ({0}) is missing. See README.md for more info.'.format(titles_path))

    df = utils.read_df_from_titles(titles_path, use_full_text=settings.FULL_TEXT)

    # save that as a csv
    df.to_csv(titles_out_path, columns=['id', 'cat1', 'cat2', 'title'], index=False)

    # normalise the title
    df['titlen'] = df['title'].apply(lambda v: utils.tokenise_title(v))

    return df


run_trials()
