"""
Processor for performing named entity tagging.
"""
import logging

from stanfordnlp.models.common import doc
from stanfordnlp.models.common.utils import unsort
from stanfordnlp.models.ner.data import DataLoader
from stanfordnlp.models.ner.trainer import Trainer
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor

logger = logging.getLogger('stanfordnlp')

class NERProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([NER])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        # set up trainer
        args = {'charlm_forward_file': config['forward_charlm_path'], 'charlm_backward_file': config['backward_charlm_path']}
        self._trainer = Trainer(args=args, model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, document):
        # set up a eval-only data loader and skip tag preprocessing
        batch = DataLoader(
            document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True, preprocess_tags=False)
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        batch.doc.set([doc.NER], [y for x in preds for y in x], to_token=True)
        # collect entities into document attribute
        total = len(batch.doc.build_ents())
        logger.debug(f'{total} entities found in document.')
        return batch.doc