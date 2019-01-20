import json
import re
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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


def process_text(steps, text, case):
    if steps is not None:
        for step in steps:
            if (step == 'lower') & (case == 'both'):
                continue
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


# plotting training and validation metrics

def plot_metrics(filename):
    with open('../logs/{0}.json'.format(filename)) as f:
        history = json.load(f)

    epochs = len(history)
    train_losses = list(map(lambda h: h['training_loss'], history))
    train_accuracies = list(map(lambda h: h['training_accuracy'], history))
    val_losses = list(map(lambda h: h['validation_loss'], history))
    val_accuracies = list(map(lambda h: h['validation_accuracy'], history))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='train_loss')
    plt.plot(range(epochs), val_losses, label='test_loss')
    plt.title('train & test loss')
    plt.grid(1)
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label='train_acc')
    plt.plot(range(epochs), val_accuracies, label='test_acc')
    plt.title('train & test accuracy')
    plt.grid(1)
    plt.xlabel('epochs')
    plt.legend()

    plt.savefig('../plots/{0}.png'.format(filename))
    

