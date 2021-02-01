from math import log10

from tensorflow.python.keras.backend import set_session
from tensorflow_addons.optimizers import AdamW

import settings
import utils
import tensorflow as tf


class Classifier:

    # The name of the pretrained model this classifier is based on.
    # Can be overridden in subclasses. Leave this blank.
    pretrained_model_name = ''

    def __init__(self, seed):
        self.seed = seed
        self.prepare_resources()

    def __del__(self):
        self.release_resources()

    def prepare_resources(self):
        '''Any preparation of resources shared by trainings.
        This is called each time a Classifier is instantiated.
        Placeholder for subclasses.'''
        pass

    def release_resources(self):
        '''release resources'''
        pass

    def get_pretrained_model_name(self):
        '''Return the name of the pretrained model this classifier
        is based on. Empty string if none.'''
        return self.pretrained_model_name.replace('/', '-')

    @classmethod
    def from_name(cls, classifier_name, seed):
        return globals()[classifier_name](seed)

    def set_datasets(self, datasets, csv_path):
        # prep sets for FT and save them on disk
        self.datasets = utils.save_ft_sets(datasets, csv_path)
        self.ds_train, self.ds_test, self.ds_valid = self.datasets
        self.df_train, self.df_test, self.df_valid = [
            ds['df'] for ds in self.datasets
        ]

    def train(self):
        raise Exception('train() must be implemented in a subclass')

    def predict(self, string):
        '''Returns [prediction, confidence]'''
        raise Exception('predict() must be implemented in a subclass')

    def get_internal_dimension(self):
        '''Returns the embeddings dimension. None if not applicable.'''
        return None

    def tokenise(self, title):
        ret = str(title)
        return ret


class FastText(Classifier):

    pretrained_model_name = settings.EMBEDDING_FILE

    def train(self):
        import fasttext

        options = {}

        if settings.VALID_PER_CLASS:
            options['autotuneValidationFile'] = self.ds_valid['path']
            options['autotuneDuration'] = settings.AUTOTUNE_DURATION
        else:
            options['epoch'] = settings.EPOCHS
            options['dim'] = settings.DIMS

        pretrained_model_name = self.get_pretrained_model_name()
        if pretrained_model_name:
            options['pretrainedVectors'] = utils.get_data_path(
                'in',
                pretrained_model_name
            )

        options['wordNgrams'] = 2
        options['lr'] = 0.2

        self.model = fasttext.train_supervised(
            self.ds_train['path'],
            **options
        )

    def predict(self, string):
        res = self.model.predict(string)

        return [utils.short_label(res[0][0]), res[1][0]]

    def get_internal_dimension(self):
        return self.model.get_dimension()

    def tokenise(self, string):
        return utils.tokenise(
            string, stop_words=True, no_numbers=True, no_small_words=True
        )


class FlairTARS(Classifier):
    '''Flair TARS few-shots training.
    It makes use of meaningful category labels found in classes.txt.
    Slow... base pretrained model (tars-base) is too heavy for our needs.
    Embeddings: 30522 x 768, 24 layers, 12 heads, 110M params.
    Flair works on top of PyTorch.
    In principle this classifier should be able to perform very well
    with ~3 samples per class. (See paper about TARS).
    '''

    pretrained_model_name = 'tars-base'

    def prepare_resources(self):
        # turn off INFO and DEBUG logging
        import flair # KEEP THIS IMPORT HERE! (it initialises 'flair' logger)
        import logging
        logger = logging.getLogger('flair')
        logger.setLevel(logging.WARNING)
        if self.seed:
            flair.set_seed(self.seed)

    def train(self):
        from flair.data import Corpus
        from flair.datasets import SentenceDataset
        from flair.data import Sentence

        self.classes = utils.read_class_titles(settings.CAT_DEPTH)
        self.classes['NOCAT'] = 'NOCAT'

        train = SentenceDataset([
            Sentence(row['titlen']).add_label(
                'law_topic',
                self.classes[row['cat1']]
            )
            for i, row
            in self.df_train.iterrows()
        ])

        # make a corpus with train and test split
        self.corpus = Corpus(train=train, dev=train)

        # 1. load base TARS
        tars = self._load_pretained_model()

        # 2. make the model aware of the desired set of labels from the new corpus
        tars.add_and_switch_to_new_task(
            "LAW_TOPIC",
            label_dictionary=self.corpus.make_label_dictionary()
        )

        # 3. initialize the text classifier trainer with your corpus
        from flair.trainers import ModelTrainer
        trainer = ModelTrainer(tars, self.corpus)

        # 4. train model
        path = settings.WORKING_DIR
        if 1:
            trainer.train(
                base_path=path,
                # path to store the model artifacts
                learning_rate=5e-2,  # 5ep, 0.2 bad; 5ep with 0.1 looks ok.
                mini_batch_size=settings.MINIBATCH,
                # mini_batch_chunk_size=1, mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
                max_epochs=settings.EPOCHS,  # terminate after 10 epochs
                train_with_dev=False,
                save_final_model=False,
                param_selection_mode=True, # True to avoid model saves
                shuffle=False, # Already done
            )

        # from flair.models.text_classification_model import TARSClassifier
        # self.model = TARSClassifier.load(
        #     os.path.join(path, 'best-model.pt')
        # )

        self.model = tars

    def predict(self, string):
        from flair.data import Sentence

        # 2. Prepare a test sentence
        sentence = Sentence(string)

        ret = ['NOCAT', 1.0]

        # 4. Predict for these classes
        self.model.predict(sentence)

        if len(sentence.labels):
            label = sentence.labels[0]
            ret = [
                self.classes.get_key_from_val(label.value),
                label.score
            ]

        return str(ret[0]), ret[1]

    def _predict_zero(self, string):
        '''Abandoned; 0-shot predictions were too poor.'''
        from flair.models.text_classification_model import TARSClassifier
        from flair.data import Sentence

        # 2. Prepare a test sentence
        sentence = Sentence(string)

        # 3. Define some classes that you want to predict using descriptive names
        ret = [len(self.classes) - 1, 1.0]

        # 4. Predict for these classes
        self._get_tars().predict_zero_shot(sentence, self.classes)

        if len(sentence.labels):
            label = sentence.labels[0]
            ret = [
                self.classes.get_key_from_val(label.value),
                label.score
            ]

        return str(ret[0]), ret[1]

    def get_internal_dimension(self):
        # return self.model.document_embeddings.embedding_length
        return None

    def _load_pretained_model(self):
        from flair.models.text_classification_model import TARSClassifier

        # 1. Load our pre-trained TARS model for English
        # Note that this must be reloaded before each training as its modified
        # during training.
        return TARSClassifier.load(self.get_pretrained_model_name())


class NaiveBayes(Classifier):

    # results are worse with tfidf, not sure why...
    use_tfidf = 0
    # not a significant difference with (1, 2)
    ngram_range = (1, 2)
    max_features = 10000

    def train(self):
        # tokenise the input
        from sklearn import feature_extraction, naive_bayes, pipeline

        if NaiveBayes.use_tfidf:
            vectorizer = feature_extraction.text.TfidfVectorizer(
                max_features=NaiveBayes.max_features,
                ngram_range=NaiveBayes.ngram_range,
                sublinear_tf=True
            )
        else:
            vectorizer = feature_extraction.text.CountVectorizer(
                max_features=NaiveBayes.max_features,
                ngram_range=NaiveBayes.ngram_range
            )

        corpus = self.df_train['titlen']

        y = self.df_train["cat1"]

        X_train = vectorizer.fit_transform(corpus)

        classifier = naive_bayes.MultinomialNB()

        self.model = pipeline.Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", classifier)]
        )

        self.model["classifier"].fit(X_train, self.df_train['cat1'])

    def predict(self, string):
        preds = self.model.predict([string])
        probs = self.model.predict_proba([string])

        return preds[0], max(probs[0])

    def tokenise(self, string):
        return utils.tokenise(
            string, stemmatise=True, lemmatise=True, stop_words=True,
            no_small_words=True, no_numbers=True
        )

class EarlyStop(tf.keras.callbacks.Callback):

    def __init__(self):
        super(EarlyStop, self).__init__()
        self.min_rate = None
        self.max_rate = None
        self.history_lr = []
        self.show_debug = True

    def get_learning_rate(self):
        return float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    def set_learning_rate(self, rate=None):
        if rate is not None:
            tf.keras.backend.set_value(self.model.optimizer.lr, rate)

    def on_epoch_begin(self, epoch, logs=None):
        self.history_lr.append(self.get_learning_rate())

    def debug(self, message):
        if self.show_debug:
            print(message)

    def pretrain(self, model, train_dataset):
        pass


class EarlyStopValAccuracy(EarlyStop):

    def __init__(self):
        super(EarlyStopValAccuracy, self).__init__()
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        if ((
                logs.get('val_accuracy', 0) > 0.99
                # if few val samples, val acc can be high accidentally
                and logs.get('accuracy') > 0.93
        )
                # stagnation & overfitting
                or logs.get('accuracy') > 0.99):
            self.model.stop_training = True

        # self.losses = []
        loss = logs.get('val_loss') or 100
        self.losses.append(loss)

        gap = 2
        if len(self.losses) > gap + 1:
            # long stagnation => stop training
            if (self.losses[-gap-2] - loss) / loss <= 0.005:
                print('\nSTAGNATING => stop')
                self.model.stop_training = True


class EarlyStopValLoss(EarlyStop):
    '''
    Learning Rate Scheduler:
        starts at very low rate: 1e-7
        then increase by <growth> (e.g. 2.0) each time
            a sequence of n batches don't show an decrease of <diff> of loss
        we note the rate for the first longer sequence to show decrease
            => self.min_rate
        we then increase that by one step <growth> and swith the shrinking mode
        in that mode we run entire epochs and only decrease rate by <shrink>
        each time there is no sufficient reduction of validation loss.
    Stopping condition:
        if last m epochs don't show a decrease of validation loss
    '''

    def __init__(self):
        super(EarlyStopValLoss, self).__init__()
        self.losses = []
        self.accs = []
        self.diff = 0.001
        self.growth = 1.1
        self.shrink = 0.2
        self.batches_per_epoch = 0
        self.rates = [1e-10, 1e-1]
        self.best = {
            'lr': None,
            'loss': 100,
            'accuracy': 0,
            'weights': None,
        }

    def pretrain(self, model, train_dataset):
        import math
        # repeat the training set into enough batches
        # to cover the total number of intermediary learning rates.
        steps = (
            math.ceil(
                math.log(self.rates[1] / self.rates[0]) /
                math.log(self.growth)
            )
        )

        self.history = model.fit(
            train_dataset.repeat().batch(settings.MINIBATCH),
            epochs=1,
            steps_per_epoch=steps,
            batch_size=settings.MINIBATCH,
            callbacks=[self],
        )

        self.set_learning_rate(self.best['lr'])
        self.model.set_weights(self.best['weights'])
        self.growth = self.shrink
        self.history_lr = []
        print(f'\nBest rate={self.best["lr"]:0.0E}')

    def on_epoch_end(self, epoch, logs={}):
        # self.losses = []
        acc = logs.get('val_loss') or 100
        self.accs.append(acc)

        if self.growth < 1:
            if len(self.accs) > 1 and ((self.accs[-2] - acc) / acc) < self.diff:
                # short stagnation => reduce LR
                lr = self.get_learning_rate()
                lr = lr * self.growth
                self.debug(f'\n shrink LR {lr:0.0E}')
                tf.keras.backend.set_value(self.model.optimizer.lr, lr)

            gap = 2
            if logs.get('val_accuracy') >= 0.99:
                self.model.stop_training = True
            elif len(self.accs) > gap + 1:
                # long stagnation => stop training
                # self.debug(f'\n accs: {self.accs} gap: {gap}')
                if (self.accs[-gap-2] - acc) / acc <= self.diff:
                    self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        self.batches_per_epoch = max(self.batches_per_epoch, batch)
        loss = logs.get('loss') or 100
        acc = logs.get('accuracy') or 0

        if self.growth > 1:
            if loss < self.best['loss'] and acc > self.best['accuracy']:
                # best LR so far
                self.best['loss'] = loss
                self.best['accuracy'] = acc
                self.best['lr'] = self.get_learning_rate()
                self.best['weights'] = self.model.get_weights()
                print(f'\n {self.best["loss"]:.3f} {self.best["lr"]:0.0E}')

        self.losses.append(loss)

    def on_train_batch_begin(self, batch, logs=None):
        if self.growth < 1:
            return

        self.model.reset_metrics()

        if batch == 0:
            self.set_learning_rate(self.rates[0])

        if len(self.losses) > 1:
            last_loss = self.losses[-1]
            lr = self.get_learning_rate()
            # self.debug(f' LR {lr:0.0E} ACC {last_loss:.3f}')
            if 0 and last_loss < self.losses[0]:
                self.debug(f'\n MAX LR {lr:0.0E}')

                # let's decrease it a bit
                lr = lr / self.growth
                # switch to shrinking mode (exploitation)
                self.growth = self.shrink

                self.debug(f' -> LR {lr:0.0E}')
            else:
                lr = lr * self.growth

            self.set_learning_rate(lr)

class Transformers(Classifier):
    '''Huggingface Transformers wrapper around Legal variant of Bert.
    Works with Tensorflow or PyTorch. Here we use Tensorflow.
    https://bit.ly/3mYGuCA
    '''

    # Couldn't make the transformer wrappers work.
    # Both should produce equivalent results.
    train_with_tensorflow_directly = 1

    # Stability of training is very sensitive to the rate.
    # 5e-5 generally works very well in <= 8 epochs
    # but occasionally get stuck forever at <0.3 acc.
    # learning_rate = 5e-5
    # TODO: use dynamic LR
    # 5e-6 more stable than e-5 but slow grinder... needs 3x more epochs.
    # BUT... 5e-6 is both too slow and not generalisable enough on title-only
    # Why? Why slower to converge on simpler inputs?
    # if settings.FULL_TEXT:
    #     learning_rate = 1e-6
    # else:
    #     learning_rate = 5e-5

    max_length = 500

    scheduler_class = EarlyStopValAccuracy
    # scheduler_class = EarlyStopValLoss

    # https://huggingface.co/transformers/pretrained_models.html
    # 6-layer, 768-hidden, 12-heads, 66M parameters
    pretrained_model_name = 'distilbert-base-uncased'
    # pretrained_model_name = 'nlpaueb/legal-bert-base-uncased'

    dtype = None
    # doesn't work on this model
    # dtype = 'float16'

    # max_epochs = settings.EPOCHS
    max_epochs = 25
    # print('FIXEPOCHS!')
    # max_epochs = 5

    def prepare_resources(self):
        # import tensorflow as tf
        # from tensorflow.compat.v1 import ConfigProto
        # config = ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        # set_session(sess)

        # from numba import cuda
        # cuda.close()
        # cuda.select_device(0)

        import os
        self.logs_path = os.path.join(settings.WORKING_DIR, 'transformers', 'logs')
        os.makedirs(self.logs_path, exist_ok=True)
        self.results_path = os.path.join(settings.WORKING_DIR, 'transformers', 'results')
        os.makedirs(self.results_path, exist_ok=True)

        # from transformers import DistilBertTokenizer as Tokenizer
        from transformers import AutoTokenizer
        Tokenizer = AutoTokenizer.from_pretrained(Transformers.pretrained_model_name)
        self.tokenizer = Tokenizer.from_pretrained(
            Transformers.pretrained_model_name
        )

        if self.dtype:
            from tensorflow import keras
            keras.backend.set_floatx(self.dtype)

            # default is 1e-7 which is too small for float16.
            # Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
            keras.backend.set_epsilon(1e-4)

    def release_resources(self):
        # https://stackoverflow.com/a/52354943
        # Without this tensorflow 2.3 crashes after 12/13 trials
        # Resource exhausted:  OOM when allocating tensor with shape[8,69,768]
        from tensorflow import keras
        keras.backend.clear_session()

    def train(self):
        import tensorflow_addons as tfa
        import tensorflow as tf

        self.scheduler = None
        # Encoding train_texts and val_texts

        # x-digit code -> unique number
        self.classes = utils.Bidict({
            c: i
            for i, c
            in enumerate(self.df_train['cat1'].unique())
        })

        # Convert datasets to transformer format
        train_dataset = self._create_transformer_dataset(self.df_train)

        val_data = train_dataset
        if settings.VALID_PER_CLASS:
            val_data = self._create_transformer_dataset(self.df_valid)
        val_data = val_data.batch(settings.MINIBATCH)

        # Fine-tune the pre-trained model
        if self.train_with_tensorflow_directly:
            from transformers import TFDistilBertForSequenceClassification

            model = TFDistilBertForSequenceClassification.from_pretrained(
                Transformers.pretrained_model_name,
                num_labels=len(self.classes)
            )
            # from transformers import TFAutoModelForPreTraining
            # model = TFAutoModelForPreTraining.from_pretrained(Transformers.pretrained_model, from_pt=True)

            # In AdamW weight decay value had no effect in our tests.
            # hard to find stable tuning, harder to beat vanilla Adam
            # optimizer = AdamW(
            #     learning_rate=0.00005,
            #     # 0.1 (default but might be too large), 0.01
            #     weight_decay=0.002
            # )
            if 1:
                learning_rate = settings.LEARNING_RATE

                if learning_rate is None:
                    if settings.FULL_TEXT:
                        # 5e-5 can fail on L1 e.g. Seed 7
                        # 1e-5 looks OK (TBC) on L1
                        # but not on L3 (no convergence)
                        # 5e-6 very slow
                        # Candidates: 5E-06 (40%), 7E-06 (49%)
                        learning_rate = 7E-06
                    else:
                        learning_rate = 5e-5

                print(f'\n LR = {learning_rate:0.0E}')
                from transformers import AdamWeightDecay
                optimizer = AdamWeightDecay(
                    learning_rate=learning_rate,
                    # 0.1 (default but might be too large), 0.01
                    # weight_decay_rate=0.0
                )
            else:
                if settings.FULL_TEXT:
                    learning_rate = 1e-3
                else:
                    learning_rate = 7e-3

                print(f'\n LR = {learning_rate:0.0E}')
                optimizer = tfa.optimizers.SGDW(
                    learning_rate=learning_rate,
                    momentum=0.0,
                    weight_decay=0.00
                    # 0.1 (default but might be too large), 0.01
                    # weight_decay_rate=0.5
                )

            # optimizer = tf.keras.optimizers.MomentumOptimizer(
            #     learning_rate=self.learning_rate
            # )
            model.compile(
                optimizer=optimizer,
                loss=model.compute_loss,
                metrics=['accuracy']
            )

            callbacks = None

            self.scheduler = None
            if self.scheduler_class:
                self.scheduler = self.scheduler_class()
                callbacks = [self.scheduler]

                self.scheduler.pretrain(model, train_dataset)

            class_weight = None
            if settings.CLASS_WEIGHT:
                class_sizes = self.df_train.cat1.value_counts()
                class_sizes_max = class_sizes.max()
                class_weight = {
                    i: class_sizes_max / class_sizes[c]
                    for c, i in self.classes.items()
                }

            self.history = model.fit(
                train_dataset.batch(settings.MINIBATCH),
                epochs=self.max_epochs,
                batch_size=settings.MINIBATCH,
                validation_data=val_data,
                callbacks=callbacks,
                class_weight=class_weight,
            )
            self.model = model

            self.render_training()
        else:
            # TODO: fix this. Almost immediate but random results
            #  compared to TF method above.
            from transformers import (
                TFTrainer, TFTrainingArguments,
                TFAutoModelForSequenceClassification
            )
            # from transformers import TFAutoModel

            training_args = TFTrainingArguments(
                output_dir=self.results_path,  # output directory
                num_train_epochs=self.max_epochs,  # total number of training epochs
                per_device_train_batch_size=settings.MINIBATCH,
                # batch size per device during training
                per_device_eval_batch_size=64,  # batch size for evaluation
                warmup_steps=500,
                # number of warmup steps for learning rate scheduler
                weight_decay=0.01,  # strength of weight decay
                logging_dir=self.logs_path,  # directory for storing logs
            )

            print('h2')

            with training_args.strategy.scope():
                # trainer_model = TFSequenceClassification.from_pretrained(
                #     Transformers.pretrained_model, num_labels=len(self.classes)
                # )
                trainer_model = TFAutoModelForSequenceClassification.from_pretrained(
                    Transformers.pretrained_model_name,
                    num_labels=len(self.classes)
                )

            print('h3')

            trainer = TFTrainer(
                model=trainer_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=train_dataset,
            )

            print('h4')

            trainer.train()

            # print(trainer.evaluate())

            self.model = trainer_model

        # trainer.evaluate()

        # Get tok & model

        # self.tokenizer = DistilBertTokenizer.from_pretrained(self.save_path)
        # self.model = TFDistilBertForSequenceClassification.from_pretrained(
        #     self.save_path)

    def render_training(self):
        history = self.history.history
        loss = [history['loss'][0]] + history['loss']
        val_loss = [history['val_loss'][0]] + history['val_loss']
        lr = [-log10(r) for r in self.scheduler.history_lr]
        lr = lr + [lr[-1]]

        import matplotlib.pyplot as plt

        epochs = range(1, len(loss)+ 1)

        plt.plot(epochs, loss, 'k', label='Training loss')
        plt.plot(epochs, val_loss, 'y', label='Validation loss')
        plt.plot(epochs, lr, 'r', label='Learning rate (1e-X)')

        min_max = [
            r or 0 for r in
            [self.scheduler.min_rate, self.scheduler.max_rate]
        ]

        plt.title(f'Training and validation loss [{min_max[0]:0.0E} - {min_max[1]:0.0E}]')
        plt.legend()

        fname = 'loss'
        plt.savefig(utils.get_data_path(
            settings.PLOT_PATH,
            utils.get_exp_key(self) + '-' + fname + '.svg')
        )
        plt.close()

    def predict(self, string):
        import tensorflow as tf

        predict_input = self.tokenizer.encode(
            string,
            truncation=True,
            padding=True,
            max_length = self.max_length,
            return_tensors="tf"
        )

        # r = (<Tensor (1,9)>, )
        r = self.model(predict_input)
        # get a Tensor (9,)
        logits = r[0][0]
        probs = tf.nn.softmax(logits).numpy()
        cat_num = probs.argmax()

        return [
            self.classes.get_key_from_val(cat_num),
            probs[cat_num]
        ]

    def tokenise(self, title):
        # not sure if this makes any difference...
        # return title.lower()
        return utils.tokenise(title, no_numbers=True)

    def _create_transformer_dataset(self, df_dataset):
        '''Returns a new transformer-compatible dataset
        from one of our datasets'''

        texts = df_dataset['title'].to_list()
        # get a list of matching categories as numbers not as codes
        labels = [
            self.classes[c]
            for c
            in df_dataset['cat1'].to_list()
        ]

        # https://stackoverflow.com/a/58667409 ?
        import numpy as np
        labels = np.asarray(labels).astype('float32')

        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length)

        # 4. Creating a Dataset object for Tensorflow
        return tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            labels
        ))

