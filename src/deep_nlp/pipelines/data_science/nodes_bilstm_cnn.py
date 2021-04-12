import torch
import sklearn.metrics as metrics
import torch.optim as optim
import torch.nn as nn
# from deep_nlp.bilstm_cnn import BilstmCnn
from deep_nlp.bilstm_cnn.bilstmcnn_gradcam import BilstmCnn
from deep_nlp.utils.early_stopping import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from mlflow import log_metric
import numpy as np
import copy


def train(cuda_allow, model, train_load, optimizer, criterion):

    epoch_loss = 0

    if cuda_allow:
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)

    model.train()

    for reviews, labels in train_load:
        if cuda_allow:
            reviews = reviews.to(torch.int64).cuda()
            labels = labels.to(torch.int64).cuda()
        else:
            reviews = reviews.to(torch.int64)
            labels = labels.to(torch.int64)

        optimizer.zero_grad()

        outputs = model(reviews)

        loss = criterion(outputs[:,1], labels.float())
        epoch_loss += loss.item()
        loss.backward()

        optimizer.step()

    return epoch_loss/len(train_load)

def evaluate(cuda_allow, model, valid_load, criterion):

    model.eval()
    correct, avg_loss, epoch_loss, total = 0, 0, 0, 0
    predictions_all, target_all, probabilities_all = [], [], []

    with torch.no_grad():
        for reviews, labels in valid_load:
            if cuda_allow:
                valid_reviews = reviews.to(torch.int64).cuda()
                valid_labels = labels.to(torch.int64).cuda()
            else :
                valid_reviews = reviews.to(torch.int64)
                valid_labels = labels.to(torch.int64)

            outputs = model(valid_reviews)
            loss_valid = criterion(outputs[:,1], valid_labels.float())

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

        avg_loss = epoch_loss / len(valid_load)
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
                                              batch_size= 1,
                                              shuffle=False)

    return train_load, valid_load, test_load

def run_train(cuda_allow, train_load, valid_load, num_epochs, patience, learning_rate, embedding_matrix
              , sentence_size, input_dim, hidden_dim, layer_dim, output_dim, feature_size, kernel_size
              , dropout_rate, padded):

    model = BilstmCnn(embedding_matrix, sentence_size, input_dim, hidden_dim, layer_dim, output_dim, feature_size
                      , kernel_size, dropout_rate, padded)
    criterion = nn.BCELoss()
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
        train_loss = train(cuda_allow, model, train_load, optimizer, criterion)

        # evaluate the model
        valid_results = evaluate(cuda_allow, model, valid_load, criterion)

        log_metric(key="Train_loss", value= train_loss, step=epoch + 1)
        log_metric(key="Validation_loss", value= valid_results["loss"], step=epoch + 1)
        log_metric(key="Accuracy", value= valid_results["accuracy"].cpu().numpy().tolist(), step=epoch + 1)
        log_metric(key= "AUC", value= valid_results["auroc"], step= epoch+1)

        if valid_results["loss"] < best_valid_loss:
            best_epoch = epoch+1

            model_clone = BilstmCnn(embedding_matrix, sentence_size, input_dim, hidden_dim, layer_dim
                                    , output_dim, feature_size, kernel_size, dropout_rate, padded)

            if cuda_allow:
                model_clone = torch.nn.DataParallel(model_clone).cuda()
            else:
                model_clone = torch.nn.DataParallel(model_clone)

            model_clone.load_state_dict(model.state_dict())

            best_model = model_clone
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
            log_metric(key="Best_train_loss", value=best_train_loss)
            log_metric(key="Best_validation_loss", value=best_valid_loss)
            log_metric(key="Best_AUC", value=best_AUC)
            log_metric(key="Best_Accuracy", value=best_accuracy.cpu().numpy().tolist())

            break

    return best_model


def save_model(model):
    return model.state_dict()


def bilstm_test(cuda_allow, embedding_matrix, sentence_size, input_dim, hidden_dim, layer_dim, output_dim
                , feature_size, kernel_size, dropout_rate, padded, test_load, model_saved, vocab, index_nothing):

    if index_nothing is None:
        index_nothing= np.array([144213])

    model = BilstmCnn(embedding_matrix, sentence_size, input_dim, hidden_dim, layer_dim, output_dim, feature_size
                      , kernel_size, dropout_rate, padded)

    if cuda_allow:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(model_saved)

    # Transform the GPU model into CPU one
    state_dict = model.module.state_dict()
    #
    cpu_model = BilstmCnn(embedding_matrix, sentence_size, input_dim, hidden_dim, layer_dim, output_dim, feature_size
                      , kernel_size, dropout_rate, padded)
    cpu_model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    correct, avg_loss, epoch_loss, total = 0, 0, 0, 0
    predictions_all, target_all, probabilities_all = [], [], []

    # Create a dict with all vocab used
    vocab_reverse = {y: x for x, y in vocab.items()}

    cpu_model.eval()
    for test_reviews, test_labels in test_load:

        test_reviews = test_reviews.to(torch.int64)
        test_labels = test_labels.to(torch.int64)

        total += test_labels.size(0)

        outputs = cpu_model(test_reviews)
        loss = criterion(outputs[:,1], test_labels.float())  # , size_average= False)

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == test_labels).sum()

        predictions_all += predicted.detach().cpu().numpy().tolist()
        probabilities_all += torch.exp(outputs)[:, 1].detach().cpu().numpy().tolist()
        target_all += test_labels.data.detach().cpu().numpy().tolist()

        # Compute GradCam for gloabal interpretability
        # Proba for class 1
        proba_1 = torch.exp(outputs)[:, 1].detach().cpu().numpy()[0]
        class_explanation = 0
        other_class_explanation = 1 if class_explanation == 0 else 0
        difference_classification = test_labels - proba_1

        print(proba_1)
        print(difference_classification[0])
        print(outputs)

        break

    avg_loss = epoch_loss / len(test_load)
    accuracy = 100 * correct / total

    # AUC and ROC Curve
    fpr, tpr, threshold = metrics.roc_curve(target_all, probabilities_all)
    auroc = metrics.auc(fpr, tpr)

    log_metric(key="Test_loss", value= avg_loss)
    log_metric(key="Accuracy_test", value= accuracy.cpu().numpy().tolist())
    log_metric(key="AUC_test", value= auroc)

    print('Résultat test : Loss: {} Accuracy: {}. AUC: {}'.format(avg_loss, accuracy, auroc))

    pass