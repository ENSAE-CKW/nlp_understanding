# define metric
import torch
import torch.optim as optim
import torch.nn as nn
import logging
from typing import Any, Dict, List, Tuple
from mlflow import log_metric
from torch.utils.data import TensorDataset

# from src.deep_nlp.embed_cnn.embcnnmodel import classifier3F
from src.deep_nlp.embed_cnn.embcnnmodel_gradcam import classifier3F

import sklearn
from sklearn import metrics
import copy

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds[:,1])
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    # initialize epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training phase
    model.train()

    for variables_x, variables_y in iterator:
        # resets the gradients after every batch
        # each batch is used in order to provide an estimation of gradient C according to the paramaeters
        optimizer.zero_grad()

        # convert to 1D tensor
        predictions = model(variables_x)

        # compute the loss
        loss = criterion(predictions[:,1], variables_y.float())

        # compute the binary accuracy
        acc = binary_accuracy(predictions, variables_y.float())

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    size = len(iterator)

    return epoch_loss / size, epoch_acc / size


def evaluate(model, iterator, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for variables_x, variables_y in iterator:
            # retrieve text and no. of words
            text = variables_x

            # convert to 1d tensor
            predictions = model(text).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions[:,1], variables_y.float())
            acc = binary_accuracy(predictions, variables_y.float())

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    size = len(iterator)

    return epoch_loss / size, epoch_acc / size

def init_model(embed,SENTENCE_SIZE, nb_filtre, type_filtre, nb_output, dropout, padded):
    type_filtre = list(type_filtre.values())
    model = classifier3F(embed, SENTENCE_SIZE, embed.shape[1], nb_filtre, type_filtre, nb_output, dropout, padded)
    model = model.float()
    return model


def creation_batch(train_data, val_data, test_data, device, batch_size):
    train_tensor_x = torch.from_numpy(train_data.drop(columns=["label"]).to_numpy()).to(device).long()
    train_tensor_y = torch.from_numpy(train_data["label"].to_numpy()).to(device).long()

    val_tensor_x = torch.from_numpy(val_data.drop(columns=["label"]).to_numpy()).to(device).long()
    val_tensor_y = torch.from_numpy(val_data["label"].to_numpy()).to(device).long()

    test_tensor_x = torch.from_numpy(test_data.drop(columns=["label"]).to_numpy()).to(device).long()
    test_tensor_y = torch.from_numpy(test_data["label"].to_numpy()).to(device).long()

    train_data = TensorDataset(train_tensor_x,train_tensor_y)
    test_data = TensorDataset(test_tensor_x, test_tensor_y)
    valid_data = TensorDataset(val_tensor_x, val_tensor_y)

    train_load = torch.utils.data.DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True)
    val_load = torch.utils.data.DataLoader(dataset= valid_data,
                                batch_size=batch_size,
                                shuffle=True)
    test_load = torch.utils.data.DataLoader(dataset= test_data,
                                batch_size=batch_size,
                                shuffle=True)
    return train_load,val_load, test_load


def run_model(model, N_EPOCHS, device, train_iterator, valid_iterator):
    best_valid_loss = float('inf')

    criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06,
                               weight_decay=0)  # weight_decay : L2 penalisation !

    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    best_acc = float('inf')
    best_train_loss = float('inf')
    best_train_acc = float('inf')

    param = model.get_params()
    best_model = classifier3F(*param)

    for epoch in range(N_EPOCHS):
        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_acc = valid_acc
            best_train_loss = train_loss
            best_train_acc = train_acc
            # best model
            best_model.load_state_dict(copy.deepcopy(model.state_dict()))
        logger = logging.getLogger(__name__)
        logger.info("Epoch %i : Accuracy : %f and Loss : %f", epoch, valid_acc,valid_loss)
        log_metric(key="Valid Accuracy", value= valid_acc)
        log_metric(key = "Valid Loss", value = valid_loss)
        log_metric(key="Train Loss", value=train_loss)
        log_metric(key="Train Accuracy", value= train_acc)


    return best_model

def save_model(model):
    return model.state_dict()


def cnn_embed_test(model, iterator, criterion, device):
    # deactivating dropout layers
    model.eval()
    model.to(device)
    #Initialisation of variables
    epoch_loss = 0
    epoch_acc = 0
    pred_test = []

    with torch.no_grad():
        for variables_x, variables_y in iterator:
            predictions = model(variables_x)
            loss = criterion(predictions[:,1], variables_y.float())
            acc = binary_accuracy(predictions, variables_y.float())
            pred_test.append(predictions)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        size = len(iterator)
        loss = epoch_loss/size
        acc = epoch_acc/size

    pred_test = torch.cat(pred_test).to(device)
    lab = [element.l for element in iterator]
    lab = torch.cat(lab).to(device)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=lab, y_score=pred_test[:, 1])
    auc = metrics.auc(fpr, tpr)
    acc = binary_accuracy(pred_test, lab)

    return auc,acc,loss