# Text Simplification and Question Generation

Text Simplification with Transformer-based model (described [here](https://github.com/Leushina/simplification)). 

The idea is to join Text Simplification and Question Generation in one model, finetuning T5 model (with huggingface 
library) on modified SQuAD dataset (by [Xinya Du et al.](https://github.com/xinyadu/nqg)). We unified source and 
target data in one line, as described in the 
[Transformer-based End-to-End Question Generation](https://arxiv.org/pdf/2005.01107v1.pdf). 

To add simplification in this End-to-End Question Generation method, we obtain simplification for contexts of the 
questions by [ACCESS](https://github.com/facebookresearch/access) library. Since End-to-End used masked language 
modeling, we needed to modify the model. Our data gas format *context - question* as a source and 
*simplfied context - question* as a target, so to use sequence-2-sequence approach, we added
masks on the question part of the source, so the model could learn how to generate it. 

## Requirements

To train we used huggingface library and seq2seq example. We modified finetune.py and utils.py for our needs, you can 
clone official [transformers repo](https://github.com/huggingface/transformers) and replace these two files with 
our versions.  To simplify context of SQuAD, take a look at Data_Preprocessing.py, you will need to install ACCESS library. 

For evaluation we use SARI and ROUGE, with [SARI library](https://github.com/XingxingZhang/pysari) 
and rouge_score library. 