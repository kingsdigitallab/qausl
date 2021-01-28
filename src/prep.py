import os
import utils
import pandas as pd
import settings
import time
from classifiers import Classifier
import utils

utils.fix_torch_cuda_long_wait()

titles_out_path = utils.get_data_path('out', 'titles.csv')


def run_trials():
    '''Run multiple trials to obtain more statistically relevant results
    and make best use of our small dataset (using cross-validation).
    For each trial, train a new classifier on a random training sample.
    '''

    catastrophic_failures = 0
    seed = None

    t0 = time.time()

    print('Load datasets')
    df = prepare_dataset()

    preds = []
    for i in range(0, settings.TRIALS):
        if settings.SAMPLE_SEED:
            seed = settings.SAMPLE_SEED + i
            utils.set_global_seed(seed)

        print('trial {}/{}{}'.format(
            i + 1,
            settings.TRIALS,
            f' ({seed} seed)' if seed else ''
        ))
        classifier_key, accuracy, df_train = train_and_test(df, preds, seed)
        if accuracy < 0.4:
            catastrophic_failures += 1
        print('-' * 40)

    t1 = time.time()

    preds = pd.DataFrame(preds)

    if 1:
        df_confusion = utils.get_confusion_matrix(preds, df_train)
        utils.render_confusion(classifier_key, df_confusion, preds)

    if 1:
        utils.render_confidence_matrix(classifier_key, preds)

    # summary - F1
    acc = len(preds.loc[preds['pred'] == preds['cat']]) / len(preds)

    conf = settings.MIN_CONFIDENCE
    positive = len(preds.loc[preds['conf'] >= conf])
    true_positive = len(preds.loc[(preds['conf'] >= conf) & (preds['pred'] == preds['cat'])])
    if positive < 1:
        positive = 0.001
        precision = 0
    else:
        precision = true_positive / positive
    recall = true_positive / len(preds)
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    if catastrophic_failures:
        catastrophic_failures = f'; {catastrophic_failures} fails.'
    else:
        catastrophic_failures = ''

    utils.log('{}; {:.2f} acc; {:.2f} prec, {:.2f} rec, {:.2f} f1 for {:.2f} conf.; {:.0f} mins.{}'.format(
        classifier_key,
        acc,
        precision,
        recall,
        f1,
        conf,
        (t1-t0)/60,
        catastrophic_failures,
    ))


def train_and_test(df, preds, seed):
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
        settings.TRAIN_PER_CLASS_MIN,
        settings.TEST_PER_CLASS,
        settings.VALID_PER_CLASS,
    )

    # TRAIN
    classifier = Classifier.from_name(settings.CLASSIFIER, seed)
    classifier.set_datasets(df, titles_out_path)
    classifier.train()

    df_test = classifier.df_test

    if settings.EVALUATE_TRAINING_SET:
        evaluate_model(classifier, classifier.df_train, display_prefix='TRAIN = ')
    accuracy = evaluate_model(classifier, df_test, preds, display_prefix='TEST  = ')
    classifier_key = utils.get_exp_key(classifier)

    classifier.release_resources()

    return classifier_key, accuracy, classifier.df_train


def prepare_dataset():
    '''Convert input .txt o .csv into a .csv file with all the necessary
    columns for training and testing classification models.'''

    # # experimental work done on first, small dataset.
    # utils.extract_transcripts_from_pdfs()
    # utils.learn_embeddings_from_transcipts()

    # load titles file into dataframe
    df_all = pd.DataFrame()
    for fileinfo in settings.DATASET_FILES:
        if not(fileinfo['can_train'] or fileinfo['can_test']):
            continue

        titles_path = utils.get_data_path('in', fileinfo['filename'])

        if not os.path.exists(titles_path):
            utils.log_error('The training file ({0}) is missing. See README.md for more info.'.format(titles_path))

        df = utils.read_df_from_titles(titles_path, use_full_text=settings.FULL_TEXT)
        for flag in ['can_train', 'can_test']:
            df[flag] = fileinfo[flag]
        df_all = df_all.append(df, ignore_index=True)

    # save that as a csv
    df_all.to_csv(
        titles_out_path,
        columns=['id', 'cat1', 'cat2', 'title', 'can_train', 'can_test'],
        index=False
    )

    # normalise the title
    classifier = Classifier.from_name(settings.CLASSIFIER, None)
    df_all['titlen'] = df_all['title'].apply(lambda v: classifier.tokenise(v))
    classifier.release_resources()

    return df_all


def evaluate_model(classifier, df_eval, preds=None, display_prefix=''):
    '''
    :param classifier: trained classifier to be evaluated
    :param df_eval: a dataframe with evaluation samples
    :param preds: optional array to be filled with evaluation predictions
    '''

    if preds is None:
        preds = []

    correct = 0
    sure = 0
    sure_correct = 0
    for idx, row in df_eval.iterrows():
        pred, confidence = classifier.predict(row['titlen'])

        preds.append({
            'cat1': row['cat1'],
            'cat2': row['cat2'],
            # cat1 or cat2, whichever is the closest to pred.
            # this is a trick to simplify analysis.
            # We consider that a match on secondary category is just as good.
            'cat': row['cat1'],
            'pred': pred,
            'conf': confidence,
            'title': row['titlen'],
        })

        if not settings.IGNORE_SECONDARY_CATEGORY:
            if pred == row['cat2']:
                preds[-1]['cat'] = row['cat2']

        comp = '<>'
        is_sure = confidence >= settings.MIN_CONFIDENCE
        if is_sure:
            sure += 1
        if pred == preds[-1]['cat']:
            correct += 1
            comp = '=='
            if is_sure:
                sure_correct += 1
        elif is_sure:
            comp = '!!'
        if comp != '==':
            if settings.SHOW_MISTAKES:
                print('{} actual: {} / {}, pred.: {} ({:.2f} c.) title: {}'.format(
                    comp,
                    row['cat1'],
                    row['cat2'],
                    pred,
                    confidence,
                    row['title'][:100].replace('\n', '')
                ))

    if sure < 1:
        sure = 0.001
        precision = 0
    else:
        precision = sure_correct / sure
    recall = sure_correct / len(df_eval)
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    accuracy = correct / len(df_eval)

    print('{}acc: {:.2f} | prec: {:.2f} of {:.2f}, {:.2f} of all, f1: {:.2f} [conf: {}] | {}d'.format(
        display_prefix,
        accuracy,
        precision,
        sure / len(df_eval),
        recall,
        f1,
        settings.MIN_CONFIDENCE,
        classifier.get_internal_dimension(),
    ))

    return accuracy

# for cap in [20, 30, 60, 100, 1000]:
# for cap in [settings.TRAIN_PER_CLASS_MAX]:
#     settings.TRAIN_PER_CLASS_MAX = cap
#     run_trials()

# seed = 34: slips... never learns

# for class_weight in [0, 1]:
#     settings.CLASS_WEIGHT = class_weight
#     for full_text in [0, 1]:
#         settings.FULL_TEXT = full_text
#         run_trials()

run_trials()
