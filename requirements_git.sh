mkdir sari
mv SARI.py sari/SARI.py
python setup.py develop
pip install rouge-score
pip install textstat
git clone https://github.com/huggingface/transformers transformers_
pip install -e ".[dev]"
pip install -r /content/transformers_/examples/requirements.txt
pip install transformers==3.5.0
(cd /content/transformers_  &&  git checkout `git rev-list -1 --before="Nov 14 2020" master`)