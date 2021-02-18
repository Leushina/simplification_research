import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sari.SARI import SARIsent
from rouge_score import rouge_scorer,scoring
from Data_Preprocessing import preprocess_QG
from TextSimp_QG_model import run_model

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

path = "/content/drive/My Drive/model/best_tfmr"
model = AutoModelForSeq2SeqLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
model.eval()


def calculate_metrics(model):
    _, _, dev = preprocess_QG()
    sentences = pd.DataFrame(dev, columns=['Complex', 'Simple'])
    sentences = sentences.groupby(['Complex']).agg(lambda x: tuple(x)).applymap(list).reset_index()

    questions = []
    with open("/content/tgt-dev.txt", 'r') as f:
        lines = f.readlines()
        for l in lines:
            questions.append(l[:-1])

    contexts = []
    with open("/content/src-dev.txt", 'r') as f:
        lines = f.readlines()
        for l in lines:
            contexts.append(l[:-1])

    filename = "/content/val.source"
    sep = '<sep>'

    with open(filename, 'r') as f:
        lines = f.readlines()
        rouge_scores = []
        sari_scores = []
        results = []
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i, line in enumerate(tqdm(lines)):
            if sep in line:
                line = line[:line.find(sep) + len(sep)]
                s = ""
                try:
                    s = run_model(line + sep)
                    s = s[s.find(sep) + len(sep):]
                except:
                    s = run_model(line + sep)
                results.append(s[0])
                ref_questions_idx = [i for i, cont in enumerate(contexts) if cont + sep == line[:-len(sep)]]
                ref_questions = [questions[i] for i in ref_questions_idx]
                fm_ = [0]
                for r in ref_questions:
                    scores = scorer.score(s, r)
                    fm_.append([round(v.fmeasure * 100, 4) for k, v in scores.items()][0])
                rouge_scores.append(max(fm_))
            else:
                s = run_model(line)[0]
                results.append(s)
                ref = list(sentences.loc[sentences['Complex'].str.contains(line[:-len(sep)])]['Simple'])
                ref = [str(r[0]) for r in ref]
                sari_scores.append(SARIsent(line, s, ref))
            if i % 10 == 0 and i != 0:
                print('Current avg rouge {}, max = {}'.format(np.mean(rouge_scores), np.max(rouge_scores)))
                k = np.argmax(rouge_scores) * 2
                print('Max rouge for context {} Result of the model == {}'.format(lines[k], results[k]))
                print('\nCurrent avg sari {}, max = {}'.format(np.mean(sari_scores), np.max(sari_scores)))
                k = np.argmax(sari_scores) * 2 + 1
                print('Max sari for context {} Result of the model == {}'.format(lines[k], results[k]))
            if i > 1000:
                break


def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    tokens = [tokenizer.decode(x) for x in res]
    return(tokens)