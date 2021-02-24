import torch
import sklearn.metrics as metrics
import torch.optim as optim
import torch.nn as nn
from deep_nlp.bilstm_cnn import BilstmCnn
from deep_nlp.utils.early_stopping import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
import numpy as np



def train(cuda_allow, model, train_load, optimizer, criterion):

    epoch_loss = 0

    model.train()

    for batch in train_load:
        if cuda_allow:
            reviews = batch[0].to(torch.int64).cuda()
            labels = batch[1].to(torch.int64).cuda()
        else:
            reviews = batch[0].to(torch.int64)
            labels = batch[1].to(torch.int64)

        optimizer.zero_grad()

        outputs = model(reviews)

        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()

        optimizer.step()

    return epoch_loss/len(train_loader)

def evaluate(cuda_allow, model, valid_load, criterion):

    model.eval()
    correct, avg_loss, epoch_loss, total = 0, 0, 0, 0
    predictions_all, target_all, probabilities_all = [], [], []

    with torch.no_grad():
        for batch in valid_load:
            if cuda_allow:
                valid_reviews = batch[0].to(torch.int64).cuda()
                valid_labels = batch[1].to(torch.int64).cuda()
            else :
                valid_reviews = batch[0].to(torch.int64)
                valid_labels = batch[1].to(torch.int64)

            outputs = model(valid_reviews)
            loss_valid = criterion(outputs, valid_labels)

            epoch_loss += loss_valid.item()

            _, predicted = torch.max(outputs.data, 1)

            predictions_all += predicted.cpu().numpy().tolist()
            probabilities_all += torch.exp(outputs)[:, 1].cpu().numpy().tolist()
            target_all += valid_labels.data.cpu().numpy().tolist()

            total += valid_labels.size(0)


            if torch.cuda.is_available():
                correct += (predicted.cpu() == valid_labels.cpu()).sum()
            else:
                correct += (predicted == valid_labels).sum()

        avg_loss = epoch_loss / len(valid_loader)
        accuracy = 100 * correct / total
        fpr, tpr, threshold = metrics.roc_curve(target_all, probabilities_all)
        auroc = metrics.auc(fpr, tpr)


    return {"loss": avg_loss, "accuracy": accuracy, "auroc": auroc}


def prepare_batch(train_data, valid_data, test_data, batch_size):

    train_load = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    valid_load = torch.utils.data.DataLoader(dataset=valid_data,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_load = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_load, valid_load, test_load

def run_train(cuda_allow, train_loader, valid_loader, num_epochs, patience, learning_rate, embedding_matrix, sentence_size, input_dim, hidden_dim, layer_dim, output_dim, feature_size, kernel_size, dropout_rate):

    model = BilstmCnn(embedding_matrix, sentence_size, input_dim, hidden_dim, layer_dim, output_dim, feature_size, kernel_size, dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    es = EarlyStopping(patience=patience)

    if cuda_allow:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    best_train_loss = 100
    best_valid_loss = 100
    best_accuracy = 0
    best_AUC = 0
    best_epoch = 0
    best_model = None

    for epoch in range(num_epochs):
        train_loss = train(cuda_allow, model, train_loader, optimizer, criterion)

        # evaluate the model
        valid_results = evaluate(cuda_allow, model, valid_loader, criterion)

        if valid_results["loss"] < best_valid_loss:
            best_epoch = epoch+1
            best_model = model
            best_train_loss = train_loss
            best_valid_loss = valid_results["loss"]
            best_accuracy = valid_results["accuracy"]
            best_AUC = valid_results["auroc"]

        print('Modèle actuel : Epoch: {}. Train loss: {}. Valid loss: {}. Accuracy: {}. AUC: {}'.format(epoch+1, train_loss, valid_results["loss"], valid_results["accuracy"],valid_results["auroc"]))
        print('Meilleur modèle : Epoch : {}. Train loss: {}. Valid loss: {}. Accuracy: {}. AUC: {}'.format(best_epoch, best_train_loss,
                                                                                        best_valid_loss,
                                                                                        best_accuracy,
                                                                                        best_AUC))

        if es.step(valid_results["loss"]):
            print("\nEarly stopping at epoch {:02}".format(epoch + 1))
            print('Meilleur modèle : Epoch : {}. Train loss: {}. Valid loss: {}. Accuracy: {}. AUC: {}'.format(best_epoch,
                                                                                                               best_train_loss,
                                                                                                               best_valid_loss,
                                                                                                               best_accuracy,
                                                                                                               best_AUC))
            break

    return best_model

def bilstm_test(model, cuda_allow, test_load) :
    model.eval()

    if cuda_allow:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    correct, avg_loss, epoch_loss, total = 0, 0, 0, 0
    predictions_all, target_all, probabilities_all = [], [], []

    with torch.no_grad():
        for test_reviews, test_labels in test_load:
            if cuda_allow:
                test_reviews = test_reviews.to(torch.int64).cuda()
                test_labels = test_labels.to(torch.int64).cuda()
            else:
                test_reviews = test_reviews.to(torch.int64)
                test_labels = test_labels.to(torch.int64)

            total += test_labels.size(0)

            outputs = model(test_reviews)
            loss = criterion(outputs, test_labels)  # , size_average= False)

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == test_labels).sum()

            predictions_all += predicted.cpu().numpy().tolist()
            probabilities_all += torch.exp(outputs)[:, 1].cpu().numpy().tolist()
            target_all += test_labels.data.cpu().numpy().tolist()

    avg_loss = epoch_loss / len(test_load)
    accuracy = 100 * correct / total

    # AUC and ROC Curve
    fpr, tpr, threshold = metrics.roc_curve(target_all, probabilities_all)
    auroc = metrics.auc(fpr, tpr)

    print('Résultat test : Loss: {} Accuracy: {}. AUC: {}'.format(avg_loss, accuracy, auroc))

    pass