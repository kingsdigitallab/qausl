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

        self.model = fasttext.train_supervised(
            self.ds_train['path'],
            **options
        )

    def predict(self, string):
        res = self.model.predict(string)

        return [res[0][0], res[1][0]]

    def get_internal_dimension(self):
        return self.model.get_dimension()
