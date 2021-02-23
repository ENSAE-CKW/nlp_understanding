import torch
import sklearn.metrics as metrics
import torch.optim as optim
import torch.nn as nn
from deep_nlp.bilstm_cnn import BilstmCnn
from deep_nlp.utils.early_stopping import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
import numpy as np



def train(cuda_allow, model, train_loader, optimizer, criterion):

    epoch_loss = 0

    model.train()

    for batch in train_loader:
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

def evaluate(cuda_allow, model, valid_loader, criterion):

    model.eval()
    correct, avg_loss, epoch_loss, total = 0, 0, 0, 0
    predictions_all, target_all, probabilities_all = [], [], []

    with torch.no_grad():
        for batch in valid_loader:
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

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, valid_loader, test_loader

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
