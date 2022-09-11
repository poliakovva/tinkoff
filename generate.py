
import argparse

import dill as pickle
import numpy as np

UNK, EOS = "_UNK_", "_EOS_"


def get_next_token(lm, prefix):
    next_tokens = lm.get_possible_next_tokens(prefix)
    tokens = list(next_tokens.keys())
    probs = list(next_tokens.values())
    next_token = np.random.choice(tokens, 1, p=probs)[0]
    return next_token


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generating text with pretrained N-gram Language Model")
    parser.add_argument('--model', required=True, help='Location of the model file')
    parser.add_argument('--prefix', required=False, help='Prefix to generate text')
    parser.add_argument('--length', required=False, type=int, help='Maximum length of text', default=10)
    args = parser.parse_args()

    with open(args.model, 'rb') as fin:
        lm = pickle.load(fin)

    if args.prefix:
        prefix = args.prefix
    else:

        prefix = np.random.choice([i[0] for i in lm.probs.keys()])

    for i in range(args.length):
        prefix += ' ' + get_next_token(lm, prefix)
        if prefix.endswith(EOS) or len(lm.get_possible_next_tokens(prefix)) == 0:
            break

    print(prefix)