import logging
from optparse import OptionParser
from typing import List

import stanza
from stanza.utils.conll import CoNLL


def parse_args():
    parser = OptionParser()
    parser.add_option('--wordvec_file', type=str, default=None, help='Word vectors filename.')
    parser.add_option('--model', type=str, default=None, help='Model file.')
    parser.add_option('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_option('--input_file', type=str, default=None, help='Input text')
    parser.add_option('--lang', type=str, help='Language')
    parser.add_option('--batch', type=int, help="number of sentences to process", default=200)
    parser.add_option('--cpu', action='store_true', help='Ignore CUDA.')
    (options, args) = parser.parse_args()
    return options


conll2str = lambda x: "\n".join(map(lambda y: "\t".join(y), x))


def parse(nlp: stanza.Pipeline, sentences: List):
    docs = nlp(sentences)
    dicts = docs.to_dict()
    conll = "\n\n".join(map(lambda x: conll2str(x), CoNLL.convert_dict(dicts)))
    return conll


if __name__ == "__main__":
    logger = logging.getLogger('stanza')
    options = parse_args()
    nlp = stanza.Pipeline(options.lang, dir=options.model, processors='tokenize,pos,lemma,depparse',
                          tokenize_no_ssplit=True) if options.model else stanza.Pipeline(options.lang,
                                                                                         processors='tokenize,pos,lemma,depparse',
                                                                                         tokenize_no_ssplit=True)

    processed = 0
    with open(options.input_file, "r") as reader, open(options.output_file, "w") as writer:
        sentences = []

        for line in reader:
            sentences.append(line.strip())
            if len(sentences) >= options.batch:
                outputs = parse(nlp, sentences)
                writer.write(outputs)
                writer.write("\n\n")
                processed += len(sentences)
                logger.info("Parsed {} sentences".format(processed))
                sentences = []

        if len(sentences) > 0:
            outputs = parse(nlp, sentences)
            writer.write(outputs)
            writer.write("\n\n")
            processed += len(sentences)
            logger.info("Parsed {} sentences".format(processed))

        logger.info("Finished")
