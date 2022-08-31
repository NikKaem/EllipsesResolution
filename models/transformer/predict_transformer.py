import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline

from models.baseline.baseline import Baseline
from utils.evaluation_metrics import Metrics


def predict_transformer():

    test_data = pd.read_csv("data/test.csv")

    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    """tokenizer.src_lang = 'de_DE'
    tokenizer.tgt_lang = 'de_DE'

    model = AutoModelForSeq2SeqLM.from_pretrained("./models/transformer/saved_models/facebook-mbart-large-50-many-to-many-mmt-10epochs-100")
    pipeline = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)

    predictions = []
    for i, sentence in enumerate(test_data.raw_sentence):
        min_length = len(tokenizer.tokenize(sentence))
        pred = pipeline(sentence, min_length=min_length, max_length=int(min_length*1.3))[0]["generated_text"]
        predictions.append(pred)
        print(str(i) + "/" + str(len(test_data.raw_sentence)))
        print(pred)

    pd.DataFrame({'input': test_data.raw_sentence, 'prediction': predictions}).to_csv('./data/predictions/transformer_predictions_best2.csv')"""

    """model = Baseline(5)
    predictions = model.predict_all(test_data.raw_sentence)"""

    predictions = test_data.raw_sentence

    metrics = Metrics(['exact_match', 'google_bleu', 'meteor', 'edit_distance'], tokenizer)
    print(metrics.compute_exact_match(test_data.full_resolution, predictions))
    print(metrics.compute_bleu(test_data.full_resolution, predictions))
    print(metrics.compute_meteor(test_data.full_resolution, predictions))
    print(metrics.compute_edit_distance(predictions, test_data.full_resolution))

    """tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    test_data = pd.read_csv("./data/predictions/transformer_predictions4.csv")
    test2 = pd.read_csv("./data/test.csv")
    metrics = Metrics(['exact_match', 'google_bleu', 'meteor'], tokenizer)
    print(metrics.compute_exact_match(test2.full_resolution, test_data.prediction))
    print(metrics.compute_bleu(test2.full_resolution, test_data.prediction))
    print(metrics.compute_meteor(test2.full_resolution, test_data.prediction))"""