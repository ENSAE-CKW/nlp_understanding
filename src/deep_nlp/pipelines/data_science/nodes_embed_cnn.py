# define metric
import torch
import torch.optim as optim
import torch.nn as nn
import logging
from typing import Any, Dict, List, Tuple
from mlflow import log_metric
from src.deep_nlp.embed_cnn.embcnnmodel import classifier3F
import sklearn
from sklearn import metrics

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

    for batch in iterator:
        # resets the gradients after every batch
        # each batch is used in order to provide an estimation of gradient C according to the paramaeters
        optimizer.zero_grad()

        # convert to 1D tensor
        predictions = model(batch[0])

        # compute the loss
        loss = criterion(predictions[:,1], batch[1].float())

        # compute the binary accuracy
        acc = binary_accuracy(predictions, batch[1].float())

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
        for batch in iterator:
            # retrieve text and no. of words
            text = batch[0]

            # convert to 1d tensor
            predictions = model(text).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions[:,1], batch[1].float())
            acc = binary_accuracy(predictions, batch[1].float())

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    size = len(iterator)

    return epoch_loss / size, epoch_acc / size

def init_model(embed,SENTENCE_SIZE, nb_filtre, type_filtre, nb_output, dropout):
    type_filtre = list(type_filtre.values())
    model = classifier3F(embed, SENTENCE_SIZE, embed.shape[1], nb_filtre, type_filtre, nb_output, dropout)
    model = model.float()
    return model


def creation_batch(train_data, val_data, test_data, device, nb_batch): #TODO : use instead torch.utils.data.DataLoader
    train_tensor_x = torch.from_numpy(train_data.drop(columns=["label"]).to_numpy()).to(device).long().split(nb_batch)
    train_tensor_y = torch.from_numpy(train_data["label"].to_numpy()).to(device).long().split(nb_batch)

    val_tensor_x = torch.from_numpy(val_data.drop(columns=["label"]).to_numpy()).to(device).long().split(nb_batch)
    val_tensor_y = torch.from_numpy(val_data["label"].to_numpy()).to(device).long().split(nb_batch)

    test_tensor_x = torch.from_numpy(test_data.drop(columns=["label"]).to_numpy()).to(device).long().split(nb_batch)
    test_tensor_y = torch.from_numpy(test_data["label"].to_numpy()).to(device).long().split(nb_batch)

    train_dict = list(zip(train_tensor_x, train_tensor_y))
    val_dict = list(zip(val_tensor_x, val_tensor_y))
    test_dict = list(zip(test_tensor_x, test_tensor_y))

    return train_dict, val_dict, test_dict


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


    return model

def cnn_embed_test(model, iterator, criterion, device):
    # deactivating dropout layers
    model.eval()
    model.to(device)
    #Initialisation of variables
    epoch_loss = 0
    epoch_acc = 0
    pred_test = []

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch[0])
            loss = criterion(predictions[:,1], batch[1].float())
            acc = binary_accuracy(predictions, batch[1].float())
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