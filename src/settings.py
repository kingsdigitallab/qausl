import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
EXPECTED_TITLE_IDS = range(1, 455)
# the seed to split into train/test sets, None if random, integer otherwise
SAMPLE_SEED = None
os.makedirs(os.path.join(DATA_PATH, 'out'), exist_ok=True)

# how many epochs for training, 500 seems ideal for 450 training set of titles
EPOCHS = 200

# dimension of the word embeddings
# fasttext default: 100; 10 still gives good results
DIMS = 100

# pre-existing embeddings file
# EMBEDDING_FILE = 'wiki-news-300d-1M-subword.vec'
# EMBEDDING_FILE = 'Law2Vec.100d.txt'
EMBEDDING_FILE = None

# 1, 2 or 3: the depth of the taxonomy for the classification
CAT_DEPTH = 3

# how many times we repeat training over different train/test splits
# 20 trials should be enough to remove variation, but long to run
TRAIN_REPEAT = 20

# minimum degree of confidence for each prediction
MIN_CONFIDENCE = 0.96

# how many test sample per class
TEST_PER_CLASS = 2

# minimum number of training samples per class
TRAIN_PER_CLASS = 1

# how many validation sample per class
# 0 means we don't use validation
VALID_PER_CLASS = 1

# in seconds
AUTOTUNE_DURATION = 30
