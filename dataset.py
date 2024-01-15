from feats.multimodal import SUPPORTED_FEATURIZERS
import logging
import numpy as np
import os
import pickle
import random
import torch
from torch.utils.data import Dataset
from utils.misc import sample

logger = logging.getLogger(__name__)


class PostKaldiConcatenationDataset(Dataset):
    def __init__(self, path, config, split, featurize=True):
        """
        :param path: Path to dataset, which should conform to the standard data format (see README file).
        :param config: Configuration dictionary under `"data"`.
        :param split: `'train'`, `'dev'`, or `'test'`.
        :param featurize: Whether to generate features.
        """
        self.path = path
        self.config = config
        self.split = split
        if featurize:
            self._featurize(self.split)
        self.egs = self._read_examples(self.split)
        if self.split == 'train':
            self.egs = self._balance_classes(self.egs)
        self.off_center = int(self.config['audio']['context'] / 2)


    def __getitem__(self, idx):
        eg = self.egs[idx]
        if self.split == 'train':
            # A training example is allowed to be missing, in which case we randomly sample another one.
            # However, a validation or test example is not allowed to be missing.
            try:
                f = open(os.path.join(self.path, 'feat', self.split, 'concat', 'egs', eg[0] + '.feat'), 'rb')
            except FileNotFoundError:
                return self.__getitem__(random.randint(0, len(self.egs) - 1))
        else:
            f = open(os.path.join(self.path, 'feat', self.split, 'concat', 'egs', eg[0] + '.feat'), 'rb')
        full = np.transpose(pickle.load(f))
        f.close()

        start = eg[1]
        end = start + 2 * self.off_center + 1  # non-inclusive
        label = eg[2]

        if start < 0:
            left_pad = np.zeros((full.shape[0], abs(start)))
            start = 0
        else:
            left_pad = np.zeros((full.shape[0], 0))

        if end > full.shape[1]:
            right_pad = np.zeros((full.shape[0], end - full.shape[1]))
            end = full.shape[1]
        else:
            right_pad = np.zeros((full.shape[0], 0))

        eg = torch.from_numpy(np.hstack((left_pad, full[:, start:end], right_pad))).type(torch.float32)
        assert eg.shape == (full.shape[0], 2 * self.off_center + 1)

        item = {'Input': eg, 'Label': label}
        if self.split != 'train':
            item['Uttword'] = (eg[0], start)
        return item

    def __len__(self):
        return len(self.egs)

    def _balance_classes(self, egs):
        """
        Balances classes by oversampling the minority classes.
        :param egs: List of examples needed to be balanced.
        :return: List of examples with balanced classes.
        """
        egs_by_class = [[] for _ in range(4)]
        for eg in egs:
            if eg[2] == 0:
                egs_by_class[0].append(eg)
            elif eg[2] == 1:
                egs_by_class[1].append(eg)
            elif eg[2] == 2:
                egs_by_class[2].append(eg)
            elif eg[2] == 3:
                egs_by_class[3].append(eg)

        most_freq = np.argmax([len(l) for l in egs_by_class])
        max_count = len(egs_by_class[most_freq])

        for i in set(range(4)) - {most_freq}:
            additional = sample(egs_by_class[i], max_count - len(egs_by_class[i]))
            egs_by_class[i].extend(additional)
        return [eg for egs_class in egs_by_class for eg in egs_class]

    def _featurize(self, split):
        self.featurizer = SUPPORTED_FEATURIZERS[self._get_featurizer_names(self.config)](self.config)
        logging.info('Featurizing ' + split + ' set')
        self.featurizer(os.path.join(self.path, split))
        raise RuntimeError('FINISHED FEATURIZING ' + split)

    def _get_featurizer_names(self, config):
        """
        Obtains featurizer names.
        :param config: Configuration dictionary under `"data"`.
        :return: A tuple in the format (audio, text), where each element is either a string or None, if the
        corresponding modality is missing.
        """
        modalities = ['audio', 'text']
        return tuple([config[m]['featurizer']['name'] if m in config else None for m in modalities])

    def _read_examples(self, split):
        with open(os.path.join(self.path, 'feat', split, 'concat', 'egs_txt', 'egs.txt'), 'r') as f:
            egs = f.read().split('\n')
        try:
            while True:
                egs.remove('')
        except ValueError:
            pass
        for i in range(len(egs)):
            eg = egs[i].split()
            eg[1] = int(eg[1])
            eg[2] = int(eg[2])
            egs[i] = eg
        return egs


class PostKaldiConcatenationTextDataset(PostKaldiConcatenationDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        input = item['Input'][:768, 150]
        item['Input'] = input
        return item


MODEL2DATASET = {
    'EfficientPunct': PostKaldiConcatenationDataset,
    'EfficientPunctBERT': PostKaldiConcatenationTextDataset,
    'EfficientPunctTDNN': PostKaldiConcatenationDataset
}