import os

# INPUTS/OUTPUTS

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
EXPECTED_TITLE_IDS = range(1, 455)
os.makedirs(os.path.join(DATA_PATH, 'out'), exist_ok=True)

TRANSCRIPTS_PATH = os.path.join(DATA_PATH, 'in', 'transcripts.txt')

# the May 2020 file
# category = 00X is saved as 00X in this CSV
TITLES_FILENAME = 'titles-455.txt'
# the Dec 2020 file
# !!! category = 00X is saved as X in this CSV
# See "Class 3" column
TITLES_FILENAME = 'titles-1398.csv'

# For temporary files during training.
# For best performance, select a local folder and files can be too big
# for remote transfer.
WORKING_DIR = '~/.qausl'
WORKING_DIR = os.path.abspath(WORKING_DIR)
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
FULL_TEXT = 0

# 1, 2 or 3: the depth of the taxonomy for the classification
CAT_DEPTH = 1

# the seed to split into train/test sets, None if random, integer otherwise
SAMPLE_SEED = None
# SAMPLE_SEED = 1

# How many times we repeat training over different train/test splits.
# 40 trials should be enough to remove variation, but long to run.
TRIALS = 2
if SAMPLE_SEED:
    TRIALS = 1

# Number of times a model sees the full training set.
# 400-500 seems ideal for 450 training set of titles.
# Note that 5 is FastText default and enough for titles in dataset 2.
# But more is needed for classification of full text.
# Generally the less data, the more epoch we need.
EPOCHS = 1

# how many test sample per class
# default 2
TEST_PER_CLASS = 2

# minimum number of training samples per class
# default 2
TRAIN_PER_CLASS = 2

# 60 for level 1, 40 level 2, 25 level 3
# We truncate majority classes to avoid blind bias towards them.
TRAIN_PER_CLASS_MAX = [0, 60, 40, 25][CAT_DEPTH]

# how many validation sample per class
# 0 means we don't use validation
# default 1 or 2
VALID_PER_CLASS = 0

# one of the class names in classifiers.py
CLASSIFIER = 'FastText'
CLASSIFIER = 'FlairTARS'

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

# if False a prediction can match the secondary category as well as the
# primary one. This will count as a true positive.
IGNORE_SECONDARY_CATEGORY = 0
