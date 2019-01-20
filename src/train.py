import os
import json
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from cnn_model import CharacterLevelCNN
from data_loader import MyDataset

from tqdm import tqdm
import utils


def train(model, training_generator, optimizer, criterion, epoch, writer, print_every=25):
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

        writer.add_scalar('Train/Loss', loss.item(), epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Train/Accuracy', training_metrics['accuracy'], epoch * num_iter_per_epoch + iter)

        if iter % print_every == 0:
            print("[Training - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                epoch + 1,
                iter + 1,
                num_iter_per_epoch,
                np.mean(losses),
                np.mean(accuraries)
            ))

    return np.mean(losses), np.mean(accuraries), losses, accuraries


def evaluate(model, validation_generator, criterion, epoch, writer, print_every=25):
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
        validation_metrics = utils.get_evaluation(labels.cpu().numpy(),
                                                  predictions.cpu().detach().numpy(),
                                                  list_metrics=["accuracy"])
        accuracy = validation_metrics['accuracy']
        losses.append(loss.item())
        accuraries.append(accuracy)

        writer.add_scalar('Test/Loss', loss.item(), epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Test/Accuracy', accuracy, epoch * num_iter_per_epoch + iter)

        if iter % print_every == 0:
            print("[Validation - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                epoch + 1,
                iter + 1,
                num_iter_per_epoch,
                np.mean(losses),
                np.mean(accuraries)))

    return np.mean(losses), np.mean(accuraries), losses, accuraries


def run(train_path, val_path, max_rows, config_path='../config.json', both_cases=False):

    time_id = int(time.time())
    log_path = '../logs/{}'.format(time_id)
    os.makedirs(log_path)
    write = SummaryWriter(log_path)

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

    training_set = MyDataset(file_path=train_path,
                             max_rows=max_rows,
                             config_path=config_path,
                             both_cases=both_cases)
    validation_set = MyDataset(file_path=val_path,
                               max_rows=max_rows,
                               config_path=config_path,
                               both_cases=both_cases)

    training_generator = DataLoader(training_set, **training_params)
    validation_generator = DataLoader(validation_set, **validation_params)

    model = CharacterLevelCNN(
        config_path='../config.json', both_cases=both_cases)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config['training']['learning_rate'], momentum=0.9)

    with open('../logs/{}.json'.format(time_id)) as f:
        json.dump([], f)

    for epoch in range(epochs):
        training_loss, training_accuracy, training_batch_losses, training_batch_accuracies = train(model,
                                                                                                   training_generator,
                                                                                                   optimizer,
                                                                                                   criterion,
                                                                                                   epoch)

        validation_loss, validation_accuracy, validation_batch_losses, validation_batch_accuracies = evaluate(model,
                                                                                                              validation_generator,
                                                                                                              criterion,
                                                                                                              epoch)
        print('[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \tval_loss: {:.4f} \tval_acc: {:.4f}'.
              format(epoch + 1, epochs, training_loss, training_accuracy, validation_loss, validation_accuracy))
        print("=" * 50)

        h = {
            'epoch': epoch,
            'training_loss': training_loss,
            'training_accuracy': training_accuracy,
            'training_batch_losses': training_batch_losses,
            'training_batch_accuraries': training_batch_accuracies,
            'validation_loss': validation_loss,
            'validation_accuracy': validation_accuracy,
            'validation_batch_losses': validation_batch_losses,
            'validation_batch_accuraries': validation_batch_accuracies
        }

        with open('../logs/{0}.json'.format(time_id), 'a') as f:
            json.dump(h, f)

        utils.plot_metrics(time_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Character Based CNN for text classification')
    parser.add_argument('--train', type=str)
    parser.add_argument('--val', type=str)
    parser.add_argument('--max_rows', type=int, default=100000)
    args = parser.parse_args()
    run(args.train, args.val, args.max_rows)
