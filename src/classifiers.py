import os

import fasttext
import settings
import utils


class Classifier:
    def __init__(self):
        pass

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

    def train(self, datasets):
        raise Exception('train() must be implemented in a subclass')

    def predict(self, string):
        '''Returns [prediction, confidence]'''
        raise Exception('predict() must be implemented in a subclass')

    def get_internal_dimension(self):
        '''Returns the embeddings dimension. None if not applicable.'''
        return None

    def tokenise(self, title):
        import re
        ret = title.lower()
        # remove small words
        ret = re.sub(r'\b\w{1,2}\b', r' ', ret)
        # remove digits
        # TODO: use D
        ret = re.sub(r'\d+', r' ', ret)
        # remove non-discriminant words
        ret = re.sub(
            r'\b(further|chap|sic|a|of|and|an|the|to|act|supplement|for|resolution|entituled|chapter|section)\b',
            '', ret)
        # remove all non-words
        ret = re.sub(r'\W+', ' ', ret)
        return ret


class FastText(Classifier):
    def train(self, datasets):
        options = {}

        if settings.VALID_PER_CLASS:
            options['autotuneValidationFile'] = self.ds_valid['path']
            options['autotuneDuration'] = settings.AUTOTUNE_DURATION
        else:
            options['epoch'] = settings.EPOCHS
            options['dim'] = settings.DIMS

        if settings.EMBEDDING_FILE:
            options['pretrainedVectors'] = utils.get_data_path(
                'in',
                settings.EMBEDDING_FILE
            )

        options['wordNgrams'] = 2

        self.model = fasttext.train_supervised(
            self.ds_train['path'],
            **options
        )

    def predict(self, string):
        res = self.model.predict(string)

        return [utils.short_label(res[0][0]), res[1][0]]

    def get_internal_dimension(self):
        return self.model.get_dimension()


class FlairTARS(Classifier):

    @classmethod
    def _get_tars(cls):
        from flair.models.text_classification_model import TARSClassifier

        ret = getattr(cls, 'tars', None)
        if ret is None:
            # 1. Load our pre-trained TARS model for English
            ret = cls.tars = TARSClassifier.load('tars-base')

        return ret

    def train(self, datasets):
        from flair.data import Corpus
        from flair.datasets import SentenceDataset
        from flair.data import Sentence

        self.classes = [
            'Personal',
            'Government',
            'Finance',
            'Law and order',
            'Religion',
            'Armed Services',
            'Social issues',
            'Economy',
            'Communications',
            'Noprediction',
        ]

        # training dataset consisting of four sentences (2 labeled as "food" and 2 labeled as "drink")
        train = SentenceDataset([
            Sentence(row['titlen']).add_label('law_topic', self.classes[int(row['cat1'])])
            for i, row
            in self.df_train.iterrows()
        ])

        # test dataset consisting of two sentences (1 labeled as "food" and 1 labeled as "drink")
        test = SentenceDataset([
            Sentence(row['titlen']).add_label('law_topic',
                                              self.classes[int(row['cat1'])])
            for i, row
            in self.df_train.iterrows()
            if i == 0
        ])

        # make a corpus with train and test split
        self.corpus = Corpus(train=train, test=test)

        # 1. load base TARS
        tars = self._get_tars()

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
                learning_rate=0.02,  # use very small learning rate
                mini_batch_size=16,
                # mini_batch_chunk_size=1,
                # small mini-batch size since corpus is tiny
                max_epochs=settings.EPOCHS,  # terminate after 10 epochs
                # train_with_dev=True,
            )

        from flair.models.text_classification_model import TARSClassifier
        self.model = TARSClassifier.load(
            os.path.join(path, 'final-model.pt')
        )

    def tokenise(self, title):
        import re
        ret = str(title)
        return ret

    def predict(self, string):
        from flair.data import Sentence

        # 2. Prepare a test sentence
        sentence = Sentence(string)

        ret = [len(self.classes) - 1, 1.0]

        # 4. Predict for these classes
        self.model.predict(sentence)

        if len(sentence.labels):
            labels = sentence.labels[0]
            ret = self.classes.index(labels.value), labels.score

        return str(ret[0]), ret[1]

    def _predict_zero(self, string):
        from flair.models.text_classification_model import TARSClassifier
        from flair.data import Sentence

        # 2. Prepare a test sentence
        sentence = Sentence(string)

        # 3. Define some classes that you want to predict using descriptive names

        ret = [len(self.classes) - 1, 1.0]

        # 4. Predict for these classes
        self._get_tars().predict_zero_shot(sentence, self.classes)

        # Print sentence with predicted labels
        # print(repr(sentence), sentence)

        if len(sentence.labels):
            labels = sentence.labels[0]
            ret = classes.index(labels.value), labels.score

        return str(ret[0]), ret[1]
