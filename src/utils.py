import re
import numpy as np
from sklearn import metrics

# text-preprocessing

def lower(text):
    return text.lower()

def remove_hashtags(text):
    clean_text = re.sub(r'#[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_user_mentions(text):
    clean_text = re.sub(r'@[A-Za-z0-9_]+', "", text)
    return clean_text


def remove_urls(text):
    clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return clean_text


preprocessing_setps = {
    'remove_hashtags': remove_hashtags,
    'remove_urls': remove_urls,
    'remove_user_mentions': remove_user_mentions,
    'lower': lower
}


def process_text(steps, text):
    if steps is not None:
        for step in steps:
            text = preprocessing_setps[step](text)

    processed_text = ""
    for tx in text:
        processed_text += tx + " "
    return processed_text

# metrics // model evaluations


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(
            metrics.confusion_matrix(y_true, y_pred))
    return output
