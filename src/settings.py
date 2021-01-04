import os

# INPUTS/OUTPUTS

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
EXPECTED_TITLE_IDS = range(1, 455)
os.makedirs(os.path.join(DATA_PATH, 'out'), exist_ok=True)

TRANSCRIPTS_PATH = os.path.join(DATA_PATH, 'in', 'transcripts.txt')

# TITLES_FILENAME = 'titles-455.txt'
# TITLES_FILENAME = 'titles-1398.csv'
# TITLES_FILENAME = 'titles-qub.csv'

DATASET_FILES = [
    # the May 2020 file from partners
    # category = 00X is saved as 00X in this CSV
    # Superseded by 'titles-1398.csv'
    {
        'filename': 'titles-455.txt',
        'can_train': 0,
        'can_test': 0,
    },
    # the Dec 2020 file from partners
    # !!! category = 00X can be encoded as X in this CSV
    # See "Class 3" column
    # It also contains full texts in a Text column.
    {
        'filename': 'titles-1398.csv',
        'can_train': 1,
        'can_test': 1,
    },
    # QUB's Irish Legal DB website, ~2363 titles
    # For comparison purpose, or training augmentation.
    {
        'filename': 'titles-qub.csv',
        'can_train': 0,
        'can_test': 0,
    },
]

# For temporary files during training.
# For best performance, select a local folder and files can be too big
# for remote transfer.
WORKING_DIR = '~/.qausl'
WORKING_DIR = os.path.expanduser(WORKING_DIR)
os.makedirs(os.path.join(WORKING_DIR), exist_ok=True)

# ---------------------------------------------------------
# EMBEDDINGS

# pre-existing embeddings file
# EMBEDDING_FILE = 'wiki-news-300d-1M-subword.vec'
EMBEDDING_FILE = 'Law2Vec.100d.txt'
# EMBEDDING_FILE = None

# dimension of the word embeddings
# fasttext default: 100; 10 still gives good results
DIMS = 100

TRANSCRIPTS_MODEL_NAME = 'transcripts.bin'
TRANSCRIPTS_MODEL_PATH = os.path.join(DATA_PATH, 'in', TRANSCRIPTS_MODEL_NAME)

PLOT_PATH = os.path.join(DATA_PATH, 'out', 'plots')
os.makedirs(PLOT_PATH, exist_ok=True)

# ---------------------------------------------------------
# GENERAL TRAINING

# If True, use the full text of a chapter instead of just its title
FULL_TEXT = 1

# 1, 2 or 3: the depth of the taxonomy for the classification
CAT_DEPTH = 3

# the seed to split into train/test sets, None if random, integer otherwise
SAMPLE_SEED = None
# SAMPLE_SEED = 1

# Number of times a model sees the full training set.
# 400-500 seems ideal for 450 training set of titles.
# Note that 5 is FastText default and enough for titles in dataset 2.
# But more is needed for classification of full text.
# Generally the less data, the more epoch we need.
EPOCHS = 5 if not FULL_TEXT else 200
EPOCHS = 8

# how many test sample per class
# default 2
TEST_PER_CLASS = 6 if CAT_DEPTH == 1 else 2

# How many times we repeat training over different train/test splits.
# 40 trials should be enough to remove variation, but long to run.
TRIALS = int(40 / TEST_PER_CLASS * 2)
if SAMPLE_SEED:
    TRIALS = 1

# minimum number of training samples per class (default 2).
# A class is 'TINY' if it contains fewer than
# VALID_PER_CLASS + TEST_PER_CLASS + TRAIN_PER_CLASS_MIN
# samples; we won't test or validate those samples.
# But we still train the model with all those samples.
TRAIN_PER_CLASS_MIN = 2

# If True, group all samples from underrepresented categories together.
# Under a special category = ###.
GROUP_TINY_CLASSES = 0

# 60 for level 1, 40 level 2, 25 level 3
# We truncate majority classes to avoid blind bias towards them.
TRAIN_PER_CLASS_MAX = [0, 60, 40, 25][CAT_DEPTH]
# TRAIN_PER_CLASS_MAX = TRAIN_PER_CLASS_MIN

# TRAIN_PER_CLASS_MAX = 100

# how many validation sample per class
# 0 means we don't use validation
# default 1 or 2
VALID_PER_CLASS = 0

# one of the class names in classifiers.py
# CLASSIFIER = 'NaiveBayes'
# CLASSIFIER = 'FastText'
# CLASSIFIER = 'FlairTARS'
CLASSIFIER = 'Transformers'

# ---------------------------------------------------------
# NN

# best with 32 (but needs GPU with enough RAM)
MINIBATCH = 8

# ---------------------------------------------------------
# FASTTEXT

# FT auto-tuning in in seconds
# FT recommends 300+, 20-60s for small dataset is ok
AUTOTUNE_DURATION = 300

# ---------------------------------------------------------
# REPORTING

REPORT_CONFIDENCES = [0, 0.5, 0.75, 0.85, 0.9, 0.95, 0.96, 0.99, 0.995, 0.999, 0.9995, 0.9999]

# minimum degree of confidence for each prediction
MIN_CONFIDENCE = 0.96
# MIN_CONFIDENCE = 0.5

# if False a prediction can match the secondary category as well as the
# primary one. This will count as a true positive.
IGNORE_SECONDARY_CATEGORY = 0

# Set this to True if your machine doesn't have a GPU.
# Avoid pytorch very long search for an absent GPU.
CPU_ONLY = 0

# True to evaluate model performance against the training set.
# It may be very slow with some classifiers.
EVALUATE_TRAINING_SET = 0

# True: report all classification mistakes on the testing set.
SHOW_MISTAKES = 1

# ---------------------------------------------------------
FEW_SHOTS = 0
if FEW_SHOTS:
    print('--- FEW_SHOTS MODE ON ---')
    MINIBATCH = 8
    TRAIN_PER_CLASS_MAX = TRAIN_PER_CLASS_MIN = 3
    TEST_PER_CLASS = 10
    EPOCHS = 5
    SAMPLE_SEED = 1
    VALID_PER_CLASS = 0
    TRIALS = 1

# ---------------------------------------------------------
# TROUBLESHOOTING
# If True runs with less demanding settings to speed up debugging or new dev.
TROUBLESHOOT = 0
if TROUBLESHOOT:
    print('!!!!!!!!!!!!!! TROUBLESHOOTING MODE ON !!!!!!!!!!!!!!')
    MINIBATCH = 4
    TRAIN_PER_CLASS_MAX = TRAIN_PER_CLASS_MIN = 1
    TEST_PER_CLASS = 1
    EPOCHS = 1
    SAMPLE_SEED = 1
    VALID_PER_CLASS = 0
    TRIALS = 1

# https://stackoverflow.com/a/38645250
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
