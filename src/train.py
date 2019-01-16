import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cnn_model import CharacterLevelCNN
from data_loader import MyDataset

import utils


def train(config_path='../config.json'):

    with open(config_path) as f:
        config = json.load(f)

    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']

    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": 0}

    validation_params = {"batch_size": batch_size,
                         "shuffle": False,
                         "num_workers": 0}

    training_set = MyDataset(config_path=config_path, train=True)
    validation_set = MyDataset(config_path=config_path, train=False)

    training_generator = DataLoader(training_set, **training_params)
    validation_generator = DataLoader(validation_set, **validation_params)

    model = CharacterLevelCNN(config_path=config_path)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    num_iter_per_epoch = len(training_generator)

    for epoch in range(epochs):
        for iter, batch in enumerate(training_generator):
            features, labels = batch
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            training_metrics = utils.get_evaluation(labels.cpu().numpy(),
                                                    predictions.cpu().detach().numpy(),
                                                    list_metrics=["accuracy"])

            print("Epoch: {}/{}, Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                epoch + 1,
                epochs,
                iter + 1,
                num_iter_per_epoch,
                loss,
                training_metrics["accuracy"]))

        model.eval()

        val_loss_list = []
        val_label_list = []
        val_pred_list = []
        for batch in validation_generator:
            val_features, val_labels = batch
            if torch.cuda.is_available():
                val_features = val_features.cuda()
                val_labels = val_labels.cuda()
            with torch.no_grad():
                val_predictions = model(val_features)
            val_loss = criterion(val_predictions, val_labels)
            val_loss_list.append(val_loss)
            val_label_list.extend(val_labels.clone().cpu())
            val_pred_list.append(val_predictions.clone().cpu())

        average_validation_loss = np.mean(val_loss_list)
        val_preds = torch.cat(val_pred_list, 0)
        val_labels = np.array(val_label_list)
        validation_metrics = utils.get_evaluation(
            val_labels, val_preds.numpy(), list_metrics=["accuracy", "confusion_matrix"])

        print("Epoch: {}/{}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            epochs,
            average_validation_loss,
            validation_metrics["accuracy"]))


if __name__ == "__main__":
    train()