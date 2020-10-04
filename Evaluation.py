import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sari.SARI import SARIsent
from Translate import translate


def get_data(path_txt="newsela_articles_20150302.aligned.sents.txt"):

    full_data_txt = pd.read_csv(path_txt, sep='\t', header=None, names=['doc', 'Vh', 'Vs', 'Complex', 'Simple'])
    full_data_txt = full_data_txt.dropna(subset=['Simple'])

    input_train, input_val, target_train, target_val = train_test_split(list(full_data_txt['Complex']),
                                                                        list(full_data_txt['Simple']),
                                                                        test_size=0.1,
                                                                        shuffle=False,
                                                                        random_state=13)
    sentences = pd.DataFrame(list(zip(input_val, target_val)),
                             columns=['Complex', 'References'])
    sentences = sentences.groupby(['Complex']).agg(lambda x: tuple(x)).applymap(list).reset_index()
    sents, refs = list(sentences['Complex']), list(sentences['References'])
    return sents, refs


def sari_boxplots():
    with open ('saris_skipgram', 'rb') as fp:
        saris_skipgram = pickle.load(fp)

    with open ('saris_glove', 'rb') as fp:
        saris_glove = pickle.load(fp)

    with open ('saris_trans', 'rb') as fp:
        saris_trans = pickle.load(fp)
    data_to_plot = [saris_glove, saris_skipgram, saris_trans]
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    # box.set( facecolor = '#1b9e77' )
    ax.set_xticklabels(['Glove', 'Word2vec', 'Transformer'])


def compute_avg_sari(model):
    sents, refs = get_data(path_txt="newsela_articles_20150302.aligned.sents.txt")
    saris = []
    n = len(sents)
    for i, pair in enumerate(zip(sents, refs)):
        sent, ref = pair
        predicted_sent = translate(model, sent)
        saris.append(SARIsent(sent, predicted_sent, ref))
    print(max(saris))
    print(sum(saris) / n)
    return max(saris), sum(saris) / n, saris


def compute_avg_stats(model, path_txt="newsela_articles_20150302.aligned.sents.txt"):
    simple_lens_words = []
    sents, refs = get_data(path_txt)
    for i, pair in enumerate(zip(sents, refs)):
        sent, ref = pair
        predicted_sent = translate(model, sent)
        simple_lens_words.append(len(predicted_sent.split()))
        if i % 500 == 0:
            print(f'Evaluated {i} sentences')

    print(sum(simple_lens_words) / len(simple_lens_words))
