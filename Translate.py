import math
import numpy as np
import spacy
from nltk.corpus import wordnet
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from Transformer_model import create_transformer

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


def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if device.type == 'cuda':
        np_mask = np_mask.cuda()
    return np_mask


def get_synonym(word, input_text):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if input_text.vocab.stoi[l.name()] != 0:
                return input_text.vocab.stoi[l.name()]

    return 0


def init_vars(src, model, max_len, k):
    init_tok = target_text.vocab.stoi['<start>']
    # src = tokenize_en(src) # .transpose(0,1)
    # sentence = Variable(torch.LongTensor([[input_text.vocab.stoi[tok] for tok in sentence]])) #.cuda()
    # src = Variable(torch.LongTensor([[input_text.vocab.stoi[tok] for tok in src]]))
    if device.type == 'cuda':
        src = src.cuda()
    src_mask = (src != input_text.vocab.stoi['<pad>']).unsqueeze(-2)
    if device.type == 'cuda':
        src_mask = src_mask.cuda()

    e_output = model.encoder(src, src_mask)

    outputs = torch.LongTensor([[init_tok]])
    if device.type == 'cuda':
        outputs = outputs.cuda()

    trg_mask = nopeak_mask(1)

    out = model.out(model.decoder(outputs,
                                  e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)

    probs, ix = out[:, -1].data.topk(k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(k, max_len).long()
    if device.type == 'cuda':
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(k, e_output.size(-2), e_output.size(-1))
    if device.type == 'cuda':
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores


def beam_search(src, model, max_len, k):
    outputs, e_outputs, log_scores = init_vars(src, model, max_len, k)
    eos_tok = target_text.vocab.stoi['<end>']
    src_mask = (src != input_text.vocab.stoi['<pad>']).unsqueeze(-2)
    if device.type == 'cuda':
        src_mask = src_mask.cuda()
    ind = None
    for i in range(2, max_len):

        trg_mask = nopeak_mask(i)
        if device.type == 'cuda':  trg_mask = trg_mask.cuda()

        out = model.out(model.decoder(outputs[:, :i],
                                      e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)

        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, k)

        ones = (outputs == eos_tok).nonzero()  # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                sentence_lengths[i] = vec[1]  # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == k:
            alpha = 0.7
            div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    if ind is None:
        length = (outputs[0] == eos_tok).nonzero()[0]
        return ' '.join([target_text.vocab.itos[tok] for tok in outputs[0][1:length]])

    else:
        length = (outputs[ind] == eos_tok).nonzero()[0]
        return ' '.join([target_text.vocab.itos[tok] for tok in outputs[ind][1:length]])


def translate_sentence(model, sentence, max_len=80, k=3):
    model.eval()
    indexed = []
    sentence = tokenize_en(sentence)
    for tok in sentence:
        if input_text.vocab.stoi[tok] != 0:
            indexed.append(input_text.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, input_text))
    sentence = Variable(torch.LongTensor([indexed]))
    if device == 0:
        sentence = sentence.cuda()

    sentence = beam_search(sentence, model, max_len, k)
    return sentence


def translate(model, src, max_len=80, custom_string=True):
    model.eval()

    if custom_string:
        src = tokenize_en(src)  # .transpose(0,1)
        # sentence = Variable(torch.LongTensor([[input_text.vocab.stoi[tok] for tok in sentence]])) #.cuda()
        if device.type == 'cuda':
            src = Variable(torch.LongTensor([[input_text.vocab.stoi[tok] for tok in src]])).cuda()
        else:
            src = Variable(torch.LongTensor([[input_text.vocab.stoi[tok] for tok in src]]))
        src_mask = (src != input_text.vocab.stoi['<pad>']).unsqueeze(-2)

    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([target_text.vocab.stoi['<start>']])
    for i in range(1, max_len):

        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype("uint8")
        if device.type == 'cuda':
            trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()
        else:
            trg_mask = Variable(torch.from_numpy(trg_mask) == 0)

        out = model.out(model.decoder(outputs[:i].unsqueeze(0),
                                      e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == target_text.vocab.stoi['<end>']:
            break
        temp = len('<start> ')
    return ' '.join([target_text.vocab.itos[ix] for ix in outputs[:i]])[temp:]


if __name__ == "__main__":

    sentence = input("Text to be simplified: \n")
    model = create_transformer()
    print(translate(model, sentence))
