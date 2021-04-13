import torch
import torch.optim as optim
import torch.nn as nn
import logging
from mlflow import log_metric
from torch.utils.data import TensorDataset
# from src.deep_nlp.embed_cnn.embcnnmodel import classifier3F
from src.deep_nlp.embed_cnn.embcnnmodel_gradcam import classifier3F
from src.deep_nlp.grad_cam.plot import plot_cm, plot_barplot
from src.deep_nlp.grad_cam.utils import preprocess_before_barplot
from src.deep_nlp.grad_cam.utils.token import order_tokens_by_importance, find_ngram

import numpy as np
import sklearn.metrics as metrics
import copy
from collections import Counter


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds[:,1])
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return np.round(acc.cpu().numpy(), 5)


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

    test_tensor_x = torch.from_numpy(test_data.drop(columns=["label"]).to_numpy()).to("cpu").long()
    test_tensor_y = torch.from_numpy(test_data["label"].to_numpy()).to("cpu").long()

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
                                batch_size= 1)
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


def cnn_embed_test(model_dict, iterator
                   , embed, SENTENCE_SIZE, nb_filtre, type_filtre, nb_output, dropout, padded
                   , vocab , class_explanation, type_map, seuil, index_nothing, embcnn_heatmap):

    if index_nothing is None:
        index_nothing= np.array([155563, 155562])

    # Load the model
    type_filtre = list(type_filtre.values())
    model= classifier3F(embed, SENTENCE_SIZE, embed.shape[1], nb_filtre, type_filtre, nb_output, dropout, padded)
    model.load_state_dict(model_dict)

    # deactivating dropout layers
    model.eval()
    model.to('cpu')

    #Initialisation of variables
    predictions_all, target_all, probabilities_all= [], [], []
    corrects, size= 0, 0
    results_one, results_two = [], []
    results_wrong_one, results_wrong_two= [], []
    bigram_token_one, bigram_token_two= [], []

    # Create a dict with all vocab used
    vocab_reverse= {y:x for x,y in vocab.items()}

    # with torch.no_grad():
    model.eval()
    i= 0
    for review, label in iterator:

        output = model(review)

        predict = torch.max(output, 1)[1].view(label.size()).data
        corrects += (predict == label.data).sum()
        size += len(label)

        predictions_all += predict.cpu().detach().numpy().tolist()
        probabilities_all += output[:, 1].cpu().detach().numpy().tolist()
        target_all += label.data.cpu().detach().numpy().tolist()

        # Model probability for class 1
        class_explanation = 0
        other_class_explanation = 1 if class_explanation == 0 else 0

        proba_1 = output[0, other_class_explanation].item()
        difference_classification = label - proba_1

        # GradCam part #
        # TODO: Manage multiclass
        # Reconstruct the sentence
        text_index = review.squeeze().numpy()
        word = np.array([vocab_reverse.get(index, "") for index in text_index])

        # if index word is in the list whe dont want, we capture its index
        if index_nothing != None: # generate warning but its ok dude
            index_nothing = np.array([])
        selected_word_bool = np.in1d(text_index, index_nothing)
        # Get index of word we want
        selected_word_index = np.where(~selected_word_bool)[0]

        # Select interesting words
        selected_word = word[selected_word_index]

        if proba_1 >= 0.5: # We classified the text as a positive review
            # So we save the gradcam for the class 1 (why the model classified it positive)
            explanations_class_two = model.get_heatmap(text=review
                                                       , num_class=other_class_explanation
                                                       , dim=[0, 2]
                                                       , type_map=type_map)[embcnn_heatmap]

            selected_explanation_two = explanations_class_two[selected_word_index]

            # Find bigram pairwise index
            best_word_explanation_index_two = np.where(explanations_class_two >= seuil)[0]
            bigram_index = find_ngram(best_word_explanation_index_two, occurence=2)
            # Generate all important bigram
            bigram_token_two += [" ".join(t) for t in
                                 [selected_word[i].tolist() for i in bigram_index]
                                 ]

            best_word_explanation_two = order_tokens_by_importance(heatmap=selected_explanation_two
                                                                   , tokens=selected_word
                                                                   , threshold=seuil)

            explications_pour_plot_two = {"mots_expli": best_word_explanation_two
                , "prob": proba_1}

            results_two.append([explications_pour_plot_two, label])

            if np.abs(difference_classification) >= seuil:
                results_wrong_two.append(results_two[-1])

        else: # same but for the class 0
            explanations_class_one = model.get_heatmap(text=review
                                                       , num_class=class_explanation
                                                       , dim=[0, 2]
                                                       , type_map=type_map)[embcnn_heatmap]

            selected_explanation_one = explanations_class_one[selected_word_index]

            # Find bigram pairwise index
            best_word_explanation_index_one = np.where(explanations_class_one >= seuil)[0]
            bigram_index = find_ngram(best_word_explanation_index_one, occurence=2)

            # Generate all important bigram
            bigram_token_one += [" ".join(t) for t in
                                 [selected_word[i].tolist() for i in bigram_index]
                                 ]

            best_word_explanation_one= order_tokens_by_importance(heatmap= selected_explanation_one
                                                                  , tokens= selected_word
                                                                  , threshold= seuil)

            explications_pour_plot_one = {"mots_expli": best_word_explanation_one
                , "prob": proba_1}

            results_one.append([explications_pour_plot_one, label])

            # If we did a big mistake (here means we predict 1 with a proba near of 1, but in fact, the true label was 0
            # So we save the result to understand why the model made a such mistake
            # The idea itsto look at the grad cam for the class the model thought the sentence was
            # Here, we save the gradcam for the class 1 because the model was really sure about is classification
            if np.abs(difference_classification) >= seuil:
                results_wrong_one.append(results_one[-1])

        i += 1
        if i % 1000 == 0:
            print(i)

        break

    fpr, tpr, threshold = metrics.roc_curve(target_all, probabilities_all)
    auroc = metrics.auc(fpr, tpr)
    accuracy = 100 * corrects / size  # avg acc per obs

    log_metric(key="Test Accuracy", value= accuracy.cpu().detach().numpy().tolist())
    log_metric(key="Test AUC", value= auroc)

    # Plotting phase (saved and mlflow artifact)
    # Confusion matrix
    plot_cm(target_all, predictions_all, path= "data/08_reporting/embed_cnn/confusion_matrix.png")

    # Global GradCam analysis
    _, _, _, mots_0_25= preprocess_before_barplot(results_one)
    mots_plus_75, _, _, _ = preprocess_before_barplot(results_two)

    #
    _, _, _, mots_0_25_wrong = preprocess_before_barplot(results_wrong_one)
    mots_plus_75_wrong, _, _, _ = preprocess_before_barplot(results_wrong_two)

    plot_barplot(mots_plus_75, path= "data/08_reporting/embed_cnn/barplot_75.png"
                 , title= "Explication globale pour la classe 1 pour les plus grosses probabilités (>= 0.75)")
    plot_barplot(mots_0_25, path= "data/08_reporting/embed_cnn/barplot_25.png"
                 , title= "Explication globale pour la classe 0 pour les plus faibles probabilités (<= 0.25)")
    plot_barplot(mots_plus_75_wrong, path="data/08_reporting/embed_cnn/barplot_75_wrong.png"
                 , title="Explication globale pour la classe 1 pour les plus grosses erreurs de prédictions")
    plot_barplot(mots_0_25_wrong, path="data/08_reporting/embed_cnn/barplot_25_wrong.png"
                 , title="Explication globale pour la classe 0 pour les plus grosses erreurs de prédictions")

    # COmpute bigram barplot
    plot_barplot(bigram_token_one, path="data/08_reporting/embed_cnn/barplot_bigram_0.png"
                 , title="Bigram pour la classe 0")
    plot_barplot(bigram_token_two, path="data/08_reporting/embed_cnn/barplot_bigram_1.png"
                 , title="Bigram pour la classe 1")

    pass
