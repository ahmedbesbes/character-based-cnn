import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cnn_model import CharacterLevelCNN
from data_loader import MyDataset

from tqdm import tqdm
import utils


def train(model, training_generator, optimizer, criterion, epoch, print_every=25):
    model.train()
    losses = []
    accuraries = []
    num_iter_per_epoch = len(training_generator)

    progress_bar = tqdm(enumerate(training_generator),
                        total=num_iter_per_epoch)

    for iter, batch in progress_bar:
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
        losses.append(loss.item())
        accuraries.append(training_metrics["accuracy"])

        if iter % print_every == 0:
            print("[Training - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                epoch + 1,
                iter + 1,
                num_iter_per_epoch,
                np.mean(losses),
                np.mean(accuraries)
                ))

    return np.mean(losses), np.mean(accuraries)


def eval(model, validation_generator, criterion, epoch, print_every=25):
    model.eval()
    losses = []
    accuraries = []
    num_iter_per_epoch = len(validation_generator)

    for iter, batch in tqdm(enumerate(validation_generator), total=num_iter_per_epoch):
        features, labels = batch
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            predictions = model(features)
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        validation_metrics = utils.get_evaluation(labels.cpu().numpy(),
                                                  predictions.cpu().detach().numpy(),
                                                  list_metrics=["accuracy"])
        accuracy = validation_metrics['accuracy']

        accuraries.append(accuracy)

        if iter % print_every == 0:
            print("[Validation - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                epoch + 1,
                iter + 1,
                num_iter_per_epoch,
                np.mean(losses),
                np.mean(accuraries)))

    return np.mean(losses), np.mean(accuraries)


def run(config_path='../config.json', both_cases=False):

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

    training_set = MyDataset(config_path=config_path, train=True, both_cases=both_cases)
    validation_set = MyDataset(config_path=config_path, train=False, both_cases=both_cases)

    training_generator = DataLoader(training_set, **training_params)
    validation_generator = DataLoader(validation_set, **validation_params)

    model = CharacterLevelCNN(config_path='../config.json', both_cases=both_cases)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config['training']['learning_rate'], momentum=0.9)

    for epoch in range(epochs):
        training_loss, training_accuracy = train(model,
                                                 training_generator,
                                                 optimizer,
                                                 criterion,
                                                 epoch)

        validation_loss, validation_accuracy = eval(model,
                                                    validation_generator,
                                                    criterion,
                                                    epoch)
        print('[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \tval_loss: {:.4f} \tval_acc: {:.4f}'.
              format(epoch+1, epochs, training_loss, training_accuracy, validation_loss, validation_accuracy))
        print("=" * 50)


if __name__ == "__main__":
    run()
