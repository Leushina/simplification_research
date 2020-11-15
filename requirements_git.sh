mkdir sari
mv SARI.py sari/SARI.py
python setup.py develop
pip install rouge-score
git clone https://github.com/huggingface/transformers transformers_
pip install -e ".[dev]"
pip install -r /content/transformers_/examples/requirements.txt
pip install -U transformers