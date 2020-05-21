import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
EXPECTED_TITLE_IDS = range(1, 455)
# the seed to split into train/test sets, None if random, integer otherwise
SAMPLE_SEED = None
os.makedirs(os.path.join(DATA_PATH, 'out'), exist_ok=True)

# how many epochs for training, 500 seems ideal for 450 training set of titles
EPOCHS = 500

# 1, 2 or 3: the depth of the taxonomy for the classification
CAT_DEPTH = 1

# minimum degree of confidence for each prediction
MIN_CONFIDENCE = 0.96
