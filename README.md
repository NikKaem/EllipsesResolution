# Resolving elliptical coordinated compound noun phrases in german medical texts

1. First setup an anaconda environment with: <code> conda create --name [env_name] python=3.8 </code>
2. Then install necessary dependencies with: <code> pip install -r requirements.txt </code>
3. Download _de_core_news_lg_ for the baseline with: <code> python -m spacy download de_core_news_lg </code>
4. Then go into _main.py_ decide whether you want to create a sweep or run a single model, add your W&B entity and project
5. Depending on sweep or not you can change _config.yaml_ or _sweep.yaml_

Note: if you just want to run an inference go to config.yaml and change _predict_, _predict_model_dir_ and _model_name_ accordingly


