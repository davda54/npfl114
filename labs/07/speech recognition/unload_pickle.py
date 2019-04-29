import pickle


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

with open('timit_mfcc.pickle', "rb") as timit_mfcc_file:
    data = pickle.load(timit_mfcc_file)

with open('timit_mfcc_train_gold.txt', 'w') as out_file:
    for sentence in data['train']['words']:
        words = intersperse(sentence, ['_'])
        words = [c for word in words for c in word]
        print(*words, sep=' ', file=out_file)
