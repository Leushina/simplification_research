import numpy as np
import pandas as pd
import json

def squad_json_to_dataframe(input_file_path='dev-v2.0.json',
                            record_path=['data', 'paragraphs', 'qas', 'answers'],
                            dev=0):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    """
    file = json.loads(open(input_file_path).read())

    # parsing different level's in the json file
    js = pd.json_normalize(file, record_path)
    m = pd.json_normalize(file, record_path[:-1])
    r = pd.json_normalize(file, record_path[:-2])

    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx

    if dev:
        main = m[['id', 'question', 'context', 'answers']].set_index('id').reset_index()
        main['c_id'] = main['context'].factorize()[0]

    else:
        ndx = np.repeat(m['id'].values, m['answers'].str.len())
        js['q_idx'] = ndx
        main = pd.concat([m[['id', 'question', 'context']].set_index('id'), js.set_index('q_idx')], 1,
                         sort=False).reset_index()
        main['c_id'] = main['context'].factorize()[0]

    return main


if __name__ == "__main__":
    record_path = ['data', 'paragraphs', 'qas', 'answers']
    dev = squad_json_to_dataframe(dev=1)
    train = squad_json_to_dataframe(input_file_path='squad2.0/train-v2.0.json')
    print(dev.head())