import settings
import utils


class Classifier:

    # The name of the pretrained model this classifier is based on.
    # Can be overridden in subclasses. Leave this blank.
    pretrained_model_name = ''

    def __init__(self):
        self.prepare_resources()

    def prepare_resources(self):
        '''Any preparation of resources shared by trainings.
        This is called each time a Classifier is instantiated.
        Placeholder for subclasses.'''
        pass

    def get_pretrained_model_name(self):
        '''Return the name of the pretrained model this classifier
        is based on. Empty string if none.'''
        return self.pretrained_model_name

    @classmethod
    def from_name(cls, classifier_name):
        return globals()[classifier_name]()

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
        if settings.SAMPLE_SEED:
            flair.set_seed(settings.SAMPLE_SEED)

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
    # 5e-6 more stable than -5 but slow grinder... needs 3x more epochs.
    learning_rate = 5e-6

    # https://huggingface.co/transformers/pretrained_models.html
    # 6-layer, 768-hidden, 12-heads, 66M parameters
    pretrained_model_name = 'distilbert-base-uncased'
    # pretrained_model = 'nlpaueb/legal-bert-base-uncased'

    def prepare_resources(self):
        import os
        self.logs_path = os.path.join(settings.WORKING_DIR, 'transformers', 'logs')
        os.makedirs(self.logs_path, exist_ok=True)
        self.results_path = os.path.join(settings.WORKING_DIR, 'transformers', 'results')
        os.makedirs(self.results_path, exist_ok=True)

        from transformers import DistilBertTokenizer as Tokenizer
        # from transformers import AutoTokenizer
        # Tokenizer = AutoTokenizer.from_pretrained(Transformers.pretrained_model)
        self.tokenizer = Tokenizer.from_pretrained(
            Transformers.pretrained_model_name
        )

    def train(self):
        import tensorflow as tf
        from transformers import (
            TFDistilBertForSequenceClassification as TFSequenceClassification,
            TFTrainer, TFTrainingArguments
        )
        from transformers import TFAutoModel

        # Encoding train_texts and val_texts

        # x-digit code -> unique number
        self.classes = utils.Bidict({
            c: i
            for i, c
            in enumerate(self.df_train['cat1'].unique())
        })

        # Convert datasets to transformer format
        train_dataset = self._create_transformer_dataset(self.df_train)

        val_data = None
        if settings.VALID_PER_CLASS:
            val_data = self._create_transformer_dataset(self.df_valid).batch(settings.MINIBATCH)

        # Fine-tune the pre-trained model
        if self.train_with_tensorflow_directly:
            model = TFSequenceClassification.from_pretrained(
                Transformers.pretrained_model_name,
                num_labels=len(self.classes)
            )
            # from transformers import TFAutoModelForPreTraining
            # model = TFAutoModelForPreTraining.from_pretrained(Transformers.pretrained_model, from_pt=True)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            )
            model.compile(
                optimizer=optimizer,
                loss=model.compute_loss,
                metrics=['accuracy']
            )

            callbacks = None
            epochs = settings.EPOCHS
            if 1:
                epochs = 15
                class EarlyStop(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs={}):
                        if (logs.get('accuracy') > 0.94):
                            self.model.stop_training = True
                callbacks = [EarlyStop()]

            model.fit(
                # train_dataset.shuffle(1000).batch(16),
                # TODO: understand what that batch() does...
                train_dataset.batch(settings.MINIBATCH),
                epochs=epochs,
                batch_size=settings.MINIBATCH,
                validation_data=val_data,
                callbacks=callbacks,
            )
            self.model = model
        else:
            # TODO: fix this. Almost immediate but random results
            #  compared to TF method above.
            training_args = TFTrainingArguments(
                output_dir=self.results_path,  # output directory
                num_train_epochs=settings.EPOCHS,  # total number of training epochs
                per_device_train_batch_size=settings.MINIBATCH,
                # batch size per device during training
                per_device_eval_batch_size=64,  # batch size for evaluation
                warmup_steps=500,
                # number of warmup steps for learning rate scheduler
                # weight_decay=0.01,  # strength of weight decay
                logging_dir=self.logs_path,  # directory for storing logs
            )

            with training_args.strategy.scope():
                # trainer_model = TFSequenceClassification.from_pretrained(
                #     Transformers.pretrained_model, num_labels=len(self.classes)
                # )
                trainer_model = TFAutoModel.from_pretrained(
                    Transformers.pretrained_model_name, num_labels=len(self.classes)
                )

            trainer = TFTrainer(
                model=trainer_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=train_dataset,
            )

            trainer.train()

            # print(trainer.evaluate())

            self.model = trainer_model

        # trainer.evaluate()

        # Get tok & model

        # self.tokenizer = DistilBertTokenizer.from_pretrained(self.save_path)
        # self.model = TFDistilBertForSequenceClassification.from_pretrained(
        #     self.save_path)

    def predict(self, string):
        import tensorflow as tf

        predict_input = self.tokenizer.encode(
            string,
            truncation=True,
            padding=True,
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
        return title.lower()

    def _create_transformer_dataset(self, df_dataset):
        '''Returns a new transformer-compatible dataset
        from one of our datasets'''
        import tensorflow as tf

        texts = df_dataset['title'].to_list()
        # get a list of matching categories as numbers not as codes
        labels = [
            self.classes[c]
            for c
            in df_dataset['cat1'].to_list()
        ]

        encodings = self.tokenizer(texts, truncation=True, padding=True)

        # 4. Creating a Dataset object for Tensorflow
        return tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            labels
        ))

