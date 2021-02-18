# define metric
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Any, Dict, List, Tuple


#define metric
from src.deep_nlp.embed_cnn.embcnnmodel import classifier3F


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

def init_model(fields, nb_filtre: int, type_filtre: Tuple[int], nb_output, dropout, SENTENCE_SIZE):
    model = classifier3F(fields["text"].vocab.vectors, SENTENCE_SIZE, fields["text"].vocab.vectors.shape[1], nb_filtre,
                         type_filtre, nb_output,
                         dropout)
    return model


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
            #torch.save(model.state_dict(), 'final_allocine.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

