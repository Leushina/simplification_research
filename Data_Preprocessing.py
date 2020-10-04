import spacy
import os
import sys
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(path_txt="newsela_articles_20150302.aligned.sents.txt"):
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
    while max_len > 270:
        max_len = max([len(sen) for sen in data['Complex']])
        data = data[data['Complex'].map(len) != max_len]

    max_len = max([len(sen) for sen in data['Simple']])
    while max_len > 170:
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
    data_txt = preprocess()
    data_txt = data_txt[['Complex', 'Simple']]

    train, val = train_test_split(data_txt, test_size=0.1, random_state=13)
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)


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
