import argparse

import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict, Counter
import dill as pickle

UNK, EOS = "_UNK_", "_EOS_"


def count_ngrams(lines: List[str], n):
    counts = defaultdict(Counter)
    # counts[(word1, word2)][word3] = how many times word3 occurred after (word1, word2)
    pfx = f"{UNK} " * (n - 1)

    for line in lines:
        tokens = (pfx + line + " " + EOS).split()
        for i in range(n, len(tokens) + 1):
            tokens = (pfx + line + " " + EOS).split()
            counts[tuple(tokens[i - n: i - 1])].update([tokens[i - 1]])
    return counts


class NGramLanguageModel:
    def __init__(self, n=3):
        """
        Train a simple count-based language model:
        compute probabilities P(w_t | prefix) given ngram counts
        """
        self.probs = defaultdict(Counter)
        self.n = n

    def fit(self, lines):
        """
        # probs[(word1, word2)][word3] = P(word3 | word1, word2)
        """
        counts = count_ngrams(lines, self.n)

        for prefix, token_counts in counts.items():
            local_sum = sum(token_counts.values())
            self.probs[prefix] = Counter({token: count / local_sum for token, count in token_counts.items()})
        return self

    def get_possible_next_tokens(self, prefix):
        """
        :param prefix: string with space-separated prefix tokens
        :returns: a dictionary {token : it's probability} for all tokens with positive probabilities
        """
        prefix = prefix.split()
        prefix = prefix[max(0, len(prefix) - self.n + 1):]
        prefix = [UNK] * (self.n - 1 - len(prefix)) + prefix
        return self.probs[tuple(prefix)]  # prefix: {word1: prob1, word2: prob2}


def get_next_token(lm, prefix):
    next_tokens = lm.get_possible_next_tokens(prefix)
    tokens = list(next_tokens.keys())
    probs = list(next_tokens.values())
    next_token = np.random.choice(tokens, 1, p=probs)[0]
    return next_token


if __name__ == '__main__':
    parser = argparse.ArgumentParser("N-gram Language Model")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Location of the data directory')
    parser.add_argument('--model', type=str, required=True,
                        help='Location of the model file')

    args = parser.parse_args()
    # data/arxivData.json
    data = pd.read_json(args.input_dir)


    def tokenize(line: str):
        """
        Write this function that tokenizes the lines
        """
        return line.split()


    lines = data.apply(lambda row: row['title'] + ' ; ' + row['summary'], axis=1).tolist()
    lines = [" ".join(tokenize(line.lower())) for line in lines]
    print("Data Loaded initializing the model")
    lm = NGramLanguageModel(n=3)
    lm.fit(lines)
    print("Model done")

    with open(args.model, 'wb') as fout:
        pickle.dump(lm, fout)
