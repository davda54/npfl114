#!/usr/bin/env python3

# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import os
import sys
import pickle
import math
import collections

import phonetics as ph
import numpy as np

def edit_distance(x, y):
    a = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(len(x) + 1): a[i][0] = i
    for j in range(len(y) + 1): a[0][j] = j
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            a[i][j] = min(
                a[i][j - 1] + 1,
                a[i - 1][j] + 1,
                a[i - 1][j - 1] + (x[i - 1] != y[j - 1])
            )
    return a[-1][-1]

vowels = {"a", "e", "i", "o", "u"}

def edit_distance_consonants(x, y):
    a = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(len(x) + 1): a[i][0] = i
    for j in range(len(y) + 1): a[0][j] = j
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            a[i][j] = min(
                a[i][j - 1] + 1,
                a[i - 1][j] + 1,
                a[i - 1][j - 1] + 1.5*((len(x) + 1 - i) / (len(x) + 1) * 0.75 + 0.25)**(1/4) * (x[i - 1] != y[j - 1]) * (0.75 if x[i - 1] in vowels or x[i - 1] in vowels else 1.0)
            )
    return a[-1][-1]

def phonetic_similarity(w1, w2):
    w1 = w1.replace("'",'')
    w2 = w2.replace("'", '')
    # print(w1, w2)
    return 0.7 * edit_distance(ph.metaphone(w1), ph.metaphone(w2)) + \
           0.3 * edit_distance(ph.soundex(w1), ph.soundex(w2))
            #0.1 * edit_distance(ph.nysiis(w1), ph.nysiis(w2)) + \

class UltimateDictionaryCleaner:

    LETTERS = [
        "<pad>", "_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ]

    MFCC_DIM = 26

    def __init__(self, golden_paths=["dev_gold.txt", "train_gold.txt"]):
        if not isinstance(golden_paths, (list,)):
            golden_paths = [golden_paths]
        self.dictionary = collections.Counter()
        for gold_path in golden_paths:
            with open(gold_path, "r", encoding="utf-8") as f:
                self._add_to_dict(f)
        print("Added {} words into the dictionary.".format(len(list(self.dictionary))))

    @staticmethod
    def _sentence_to_words(sentence):
        words = []
        chars = sentence.split(" ")
        under_indices = [-1] + [i for i, v in enumerate(chars) if v == '_'] + [len(chars)]
        for start, end in zip(under_indices, under_indices[1:]):
            words.append(''.join(chars[(start+1):end]))
        return words

    def _add_to_dict(self, file):
        sentences = [line.rstrip("\n") for line in file]
        for sentence in sentences:
            self.dictionary.update(self._sentence_to_words(sentence))

    def clean(self, similarity_threshold=1.0, frequency_threshold=5, test_path="test_predict.txt", output_path="test_predict_cleaned.txt"):
        dictionary_words = [k for k, v in self.dictionary.items() if v > frequency_threshold]
        with open(test_path, "r", encoding="utf-8") as inf:
            sentences = [line.rstrip("\n") for line in inf]
        sentences_with_words = []
        unique_predict_words = set()
        for sentence in sentences:
            words = self._sentence_to_words(sentence)
            sentences_with_words.append(words)
            unique_predict_words.update(words)
        mapping = {}
        for i, predict_word in enumerate(unique_predict_words):
            print(f'tried to map {i+1}/{len(unique_predict_words)} words', flush=True)
            if len(predict_word) < 3: continue
            log_length = math.log(len(predict_word))
            for dict_word in dictionary_words:
                distance = edit_distance_consonants(predict_word, dict_word) / log_length
                # distance = phonetic_similarity(predict_word, dict_word)
                if distance < similarity_threshold:
                    if predict_word in mapping and mapping[predict_word][0] < distance: continue
                    mapping[predict_word] = (distance, dict_word)
        with open(output_path, "w", encoding="utf-8") as ouf:
            for sentence in sentences_with_words:
                new_sentence = []
                for word in sentence:
                    if word == '': continue
                    if len(word) > 2 and word in mapping: new_sentence.append(" ".join(mapping[word][1]))
                    else: new_sentence.append(" ".join(word))
                print(" _ ".join(new_sentence), file=ouf)


if __name__ == "__main__":
    cleaner = UltimateDictionaryCleaner(["train_gold.txt", "dev_gold.txt"])
    cleaner.clean(test_path="test_predict_2.txt", output_path="test_predict_cleaned.txt")

    with open("dev_predict_cleaned.txt", "r", encoding="utf-8") as system_file:
        system = [line.rstrip("\n") for line in system_file]

    with open("dev_gold.txt", "r", encoding="utf-8") as gold_file:
        gold = [line.rstrip("\n") for line in gold_file]

    if len(system) < len(gold):
        raise RuntimeError("The system output is shorter than gold data: {} vs {}.".format(len(system), len(gold)))

    score = 0
    for i in range(len(gold)):
        gold_sentence = gold[i].split(" ")
        system_sentence = system[i].split(" ")
        score += edit_distance(gold_sentence, system_sentence) / len(gold_sentence)

    #   not processed ---> 28.47%
    #   black magic ugly rubbish ---> 28.05
    #   1.5 ---> 27.25
    #   1.75 --> 27.30
    #   2.25 --> 27.30

    print("Average normalized edit distance: {:.2f}%".format(100 * score / len(gold)))

