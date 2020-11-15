import spacy
import os
import sys
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import fileinput
from access.preprocessors import get_preprocessors
from access.resources.prepare import prepare_models
from access.simplifiers import get_fairseq_simplifier, get_preprocessed_simplifier
from access.text import word_tokenize
from access.utils.helpers import yield_lines, write_lines, get_temp_filepath, mute

def preprocess(path_txt="newsela_articles_20150302.aligned.sents.txt", max_len_accep=300):
    """
    Deleting rows without simplification, rows with too long sentences
    creating pandas dataframe
    :param path_txt: path to the dataset
    :return: preprocessed dataframe
    """
    path_txt = os.path.join(sys.path[0], path_txt)
    data = pd.read_csv(path_txt, sep='\t', header=None, names=['doc', 'Vh', 'Vs', 'Complex', 'Simple'])
    data = data.dropna(subset=['Simple'])
    # data.drop_duplicates(subset='Complex', keep='first')
    # print("Mean length of the input sentences is {:.3f}".format(
    #     sum([len(sen) for sen in data['Complex']]) / len(data)))
    # print("Mean length of the output sentences is {:.3f}".format(sum([len(sen) for sen in data['Simple'] \
    #                                                                   if type(sen) == str]) / len(data)))
    max_len = max([len(sen) for sen in data['Complex']])
    while max_len > max_len_accep:
        max_len = max([len(sen) for sen in data['Complex']])
        data = data[data['Complex'].map(len) != max_len]

    max_len = max([len(sen) for sen in data['Simple']])
    while max_len > max_len_accep:
        max_len = max([len(sen) for sen in data['Simple']])
        data = data[data['Simple'].map(len) != max_len]
    # print("\nTotal count of samples - ", len(data))

    # # show lens in words and in chars of sentences
    # lists_chars, list_words = data_distribution(data)
    # list_words.hist(bins=30)
    # plt.show()
    # lists_chars.hist(bins=30)
    # plt.show()

    return data


def create_csv():
    data_txt = preprocess(max_len_accep=270)
    data_txt = data_txt[['Complex', 'Simple']]

    train, val = train_test_split(data_txt, test_size=0.1, random_state=13)
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)


def preprocess_QG():
    data_txt = preprocess()
    data_txt = data_txt[['Complex', 'Simple']]

    data_txt = data_txt[~data_txt['Complex'].str.contains('\n')]
    data_txt = data_txt[~data_txt['Simple'].str.contains('\n')]

    train, val = train_test_split(data_txt, test_size=0.33, shuffle=False)

    # for testing we don't need same complex sentences and different simplifications
    # only different complex sentences
    val = val.drop_duplicates(subset='Complex', keep='first')

    # since we don't shuffle, delete same input sentences
    # from train and val if there is such sentences
    val = val[~val['Complex'].isin(train['Complex'])]
    test, dev = train_test_split(val, test_size=0.47, shuffle=False)

    return train, test, dev


def create_mixed_QG(sep=' <sep>', sos='<sos> ', eos=' <eos>', dir='/content/data'):
    train, test, dev = preprocess_QG()

    for split in ['dev', 'test', 'train']:
        dict_ = {
            'src-': [],
            'tgt-': [],
            'simp-': []
        }
        for key in dict_:
            filename = key + split + '.txt'
            with open(os.path.join(dir, filename), 'r') as f:
                # questions for sentences
                lines = f.readlines()
                for line in lines:
                    dict_[key].append(line[:-1])

        assert len(dict_['src-']) == len(dict_['tgt-']) == len(dict_['simp-']), \
            print(len(dict_['src-']), len(dict_['tgt-']), len(dict_['simp-']))

        # data for source - context + question
        source_data = []
        # data for target - simple sent + question
        target_data = []

        # src = list(eval(split)['Complex'])
        # trg = list(eval(split)['Simple'])
        # assert len(src) == len(trg), print(len(src), len(trg))

        for i, sent in enumerate(dict_['src-']):
            # <eos> and <sos> is dropped
            if len(sent + sep + " " + dict_['tgt-'][i] + '\n') < 300:
                source_data.append(dict_['src-'][i] + sep + " " + dict_['tgt-'][i] + '\n')
                target_data.append(dict_['simp-'][i] + sep + " " + dict_['tgt-'][i] + '\n')
                # if i < len(src):
                #   source_data.append(src[i]  + '\n')
                #   target_data.append( trg[i]  + '\n')

        # # if we don't mix questions and simplifications
        # source_data.extend([str(sos + s + eos + '\n') for s in src])
        # target_data.extend([str(sos + s + eos + '\n') for s in trg])

        print("Maximum len of sentence in source for {} = {} ".format(split,
                                                                      max([len(s) for s in source_data])))

        print("Maximum len of sentence in target for {} = {} ".format(split,
                                                                      max([len(s) for s in target_data])))
        for l in [source_data, target_data]:
            l[-1] = l[-1][:-1]
        assert len(source_data) == len(target_data), \
            print(len(source_data), len(target_data))

        if not os.path.isfile(dir + '/new_data/' + split + '.source.txt'):
            with open(dir + '/new_data/' + split + '.source.txt', 'w') as f:
                for sent in source_data:
                    # context + question / complex sentence
                    f.write(sent)
        if not os.path.isfile(dir + '/new_data/' + split + '.target.txt'):
            print(len(target_data))
            with open(dir + '/new_data/' + split + '.target.txt', 'w') as f:
                for sent in target_data:
                    # simplified context + question / simple sentence
                    f.write(sent)
        return source_data, target_data


def data_stats():
    source_data, target_data = create_mixed_QG()

    print("Mean length of the input sentences is {:.3f}".format(
        sum([len(sen) for sen in source_data]) / len(source_data)))
    print("Mean length of the output sentences is {:.3f}".format(
        sum([len(sen) for sen in target_data]) / len(target_data)))
    # empty lists
    complex_lens_words = []
    simple_lens_words = []

    complex_lens = []
    simple_lens = []

    # populate the lists with sentence lengths
    for i in source_data:
        complex_lens_words.append(len(i.split()))
        complex_lens.append(len(i))

    for i in target_data:
        simple_lens_words.append(len(i.split()))
        simple_lens.append(len(i))

    length_words_df = pd.DataFrame({'complex': complex_lens_words, 'simple': simple_lens_words})
    length_df = pd.DataFrame({'complex': complex_lens, 'simple': simple_lens})

    print("Mean length of the input sentences is {:.3f}".format(
        sum([len(sen.split()) for sen in source_data]) / len(source_data)))
    print("Mean length of the output sentences is {:.3f}".format(sum([len(sen.split()) for sen in target_data \
                                                                      if type(sen) == str]) / len(target_data)))
    length_words_df.hist(bins=30)
    plt.show()
    length_df.hist(bins=30)
    plt.show()

    data_to_plot = [complex_lens_words, simple_lens_words]
    fig = plt.figure(1, figsize=(6, 4))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    # bp = ax.boxplot(length_words_df)
    # box.set( facecolor = '#1b9e77' )
    ax.set_xticklabels(['Complex', 'Simplified'])


def simplify_sents(dir='/content/data', split='train'):

    best_model_dir = prepare_models()
    recommended_preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': 0.95},
        'LevenshteinPreprocessor': {'target_ratio': 0.75},
        'WordRankRatioPreprocessor': {'target_ratio': 0.75},
        'SentencePiecePreprocessor': {'vocab_size': 10000},
    }
    preprocessors = get_preprocessors(recommended_preprocessors_kwargs)
    simplifier = get_fairseq_simplifier(best_model_dir, beam=8)
    simplifier = get_preprocessed_simplifier(simplifier, preprocessors=preprocessors)

    result = []
    filename = 'src-' + split + '.txt'
    # train files too big, need to split them and simplify individually
    sents = []
    with open(os.path.join(dir, filename), 'r') as f:
        # all contexts for questions
        lines = f.readlines()
        for line in lines:
            sents.append(line[:-1])

    # split src-train on smaller files ## start
    filenames_ = []
    j = 0  # 60000
    for i, sent in enumerate(sents):
        if i % 20000 == 0:  # and i > 60000:
            filenames_.append('src-' + split + str(i) + '.txt')
            with open(os.path.join(dir, filenames_[-1]), 'w') as f:
                for k in range(j, i):
                    f.write(sents[k] + '\n')
            j = i
    filenames_.append('src-' + split + str(i) + '.txt')
    with open(os.path.join(dir, filenames_[-1]), 'w') as f:
        for k in range(j, len(sents)):
            f.write(sents[k] + '\n')
    print('file splitted to smaller ones')
    # split src-train on smaller files ## end

    print('Simplification of {} started'.format(filename))
    # simplified sentences for questions
    for fname in filenames_:
        out_filepath = 'simp-' + fname  ## file from newsela
        if os.path.isfile(out_filepath):
            # # if already simplified, read files
            # with open(out_filepath, 'r') as f:
            #     lines = f.readlines()
            #     for line in lines:
            #         dict_[key].append(line[:-1])
            print('read from file')
        else:
            # if not simplified yet
            print('simplifying {}'.format(fname))
            with mute():
                simplifier(os.path.join(dir, fname), out_filepath)
            for line in yield_lines(out_filepath):
                result.append(line)
    return result
    # os.remove(pred_filepath)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        spacy.prefer_gpu()
    en = spacy.load('en_core_web_sm')

    # input_text.build_vocab(train, val)
    # target_text.build_vocab(train, val)
    #
    # torch.save(input_text, 'input_text.p')
    # torch.save(target_text, 'target_text.p')

    input_text = torch.load('input_text.p')
    target_text = torch.load('target_text.p')
