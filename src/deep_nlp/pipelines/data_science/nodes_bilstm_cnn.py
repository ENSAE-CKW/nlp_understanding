import torch
import sklearn.metrics as metrics
import torch.optim as optim
import torch.nn as nn
# from deep_nlp.bilstm_cnn import BilstmCnn
from deep_nlp.bilstm_cnn.bilstmcnn_gradcam import BilstmCnn
from deep_nlp.utils.early_stopping import EarlyStopping
from src.deep_nlp.grad_cam.utils.token import order_tokens_by_importance, find_ngram
from src.deep_nlp.grad_cam.plot import plot_cm, plot_barplot
from src.deep_nlp.grad_cam.utils import preprocess_before_barplot

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

            predict = torch.max(outputs, 1)[1].view(valid_labels.size()).data


            predictions_all += predict.cpu().numpy().tolist()
            probabilities_all += outputs[:, 1].cpu().numpy().tolist()
            target_all += valid_labels.data.cpu().numpy().tolist()

            total += valid_labels.size(0)


            if torch.cuda.is_available():
                correct += (predict.cpu() == valid_labels.cpu()).sum()
            else:
                correct += (predict == valid_labels).sum()

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
                , feature_size, kernel_size, dropout_rate, padded, test_load, model_saved, vocab, index_nothing
                , type_map, seuil):

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

    criterion = nn.BCELoss()
    correct, avg_loss, epoch_loss, total = 0, 0, 0, 0
    predictions_all, target_all, probabilities_all = [], [], []
    bigram_token_two, bigram_token_one= [], []
    results_two, results_one= [], []
    results_wrong_two, results_wrong_one= [], []

    # Create a dict with all vocab used
    vocab_reverse = {y: x for x, y in vocab.items()}

    cpu_model.eval()
    i= 0
    for test_reviews, test_labels in test_load:

        test_reviews = test_reviews.to(torch.int64)
        test_labels = test_labels.to(torch.int64)

        total += test_labels.size(0)

        outputs = cpu_model(test_reviews)
        loss = criterion(outputs[:,1], test_labels.float())  # , size_average= False)

        epoch_loss += loss.item()
        predict = torch.max(outputs, 1)[1].view(test_labels.size()).data
        correct += (predict == test_labels).sum()

        predictions_all += predict.detach().cpu().numpy().tolist()
        probabilities_all += outputs[:, 1].detach().cpu().numpy().tolist()
        target_all += test_labels.data.detach().cpu().numpy().tolist()

        # Compute GradCam for gloabal interpretability
        # Proba for class 1
        proba_1 = outputs[:, 1].detach().cpu().numpy()[0]
        class_explanation = 0
        other_class_explanation = 1 if class_explanation == 0 else 0
        difference_classification = test_labels - proba_1

        # GradCam part #
        # Reconstruct the sentence
        text_index = test_reviews.squeeze().numpy()
        word = np.array([vocab_reverse.get(index, "") for index in text_index])
         # TODO dont why there is black value counted at the end ?

        # if index word is in the list whe dont want, we capture its index
        if index_nothing != None:  # generate warning but its ok dude
            index_nothing = np.array([])
        selected_word_bool = np.in1d(text_index, index_nothing)
        # Get index of word we want
        selected_word_index = np.where(~selected_word_bool)[0]

        # Select interesting words
        selected_word = word[selected_word_index]

        if proba_1 >= 0.5: # We classified the text as a positive review
            # So we save the gradcam for the class 1 (why the model classified it positive)
            explanations_class_two = cpu_model.get_heatmap(text= test_reviews
                                                       , num_class=other_class_explanation
                                                       , dim=[0, 2]
                                                       , type_map=type_map)[-1] # only one type of heatmap
                                                                                # on this model

            selected_explanation_two = explanations_class_two[selected_word_index]

            # Find bigram pairwise index
            best_word_explanation_index_two = np.where(explanations_class_two >= seuil)[0]
            bigram_index = find_ngram(best_word_explanation_index_two, occurence=2)
            # Generate all important bigram
            bigram_token_two += [" ".join(t) for t in
                                 [selected_word[i].tolist() for i in bigram_index]
                                 if t not in [" ", ""]
                                 ]
            bigram_token_two= [t for t in bigram_token_two if t[-1] not in [" ", ""]] # if last char is a space
            # delete it because it's a lonely word

            best_word_explanation_two = order_tokens_by_importance(heatmap=selected_explanation_two
                                                                   , tokens=selected_word
                                                                   , threshold=seuil)

            explications_pour_plot_two = {"mots_expli": [t for t in best_word_explanation_two if t not in [" ", ""]]
                , "prob": proba_1}

            results_two.append([explications_pour_plot_two, test_labels.data])

            if np.abs(difference_classification) >= seuil:
                results_wrong_two.append(results_two[-1])

        else: # same but for the class 0
            explanations_class_one = cpu_model.get_heatmap(text= test_reviews
                                                       , num_class=class_explanation
                                                       , dim=[0, 2]
                                                       , type_map=type_map)[-1]

            selected_explanation_one = explanations_class_one[selected_word_index]

            # Find bigram pairwise index
            best_word_explanation_index_one = np.where(explanations_class_one >= seuil)[0]
            bigram_index = find_ngram(best_word_explanation_index_one, occurence=2)

            # TODO: delte phantom token before get_heatmap (for norm)
            # Generate all important bigram
            bigram_token_one += [" ".join(t) for t in
                                 [selected_word[i].tolist() for i in bigram_index]
                                 ]
            bigram_token_one = [t for t in bigram_token_one if t[-1] not in [" ", ""]]

            best_word_explanation_one= order_tokens_by_importance(heatmap= selected_explanation_one
                                                                  , tokens= selected_word
                                                                  , threshold= seuil)

            explications_pour_plot_one = {"mots_expli": [t for t in best_word_explanation_one if t not in [" ", ""]]
                , "prob": proba_1}

            results_one.append([explications_pour_plot_one, test_labels.data])

            # If we did a big mistake (here means we predict 1 with a proba near of 1, but in fact, the true label was 0
            # So we save the result to understand why the model made a such mistake
            # The idea itsto look at the grad cam for the class the model thought the sentence was
            # Here, we save the gradcam for the class 1 because the model was really sure about is classification
            if np.abs(difference_classification) >= seuil:
                results_wrong_one.append(results_one[-1])

        i += 1
        if i % 1000 == 0:
            print(i)

    avg_loss = epoch_loss / len(test_load)
    accuracy = 100 * correct / total

    # AUC and ROC Curve
    fpr, tpr, threshold = metrics.roc_curve(target_all, probabilities_all)
    auroc = metrics.auc(fpr, tpr)

    log_metric(key="Test_loss", value= avg_loss)
    log_metric(key="Accuracy_test", value= accuracy.cpu().numpy().tolist())
    log_metric(key="AUC_test", value= auroc)

    print('Résultat test : Loss: {} Accuracy: {}. AUC: {}'.format(avg_loss, accuracy, auroc))

    # Plotting phase (saved and mlflow artifact)
    # Confusion matrix
    plot_cm(target_all, predictions_all, path="data/08_reporting/bilstm_cnn/confusion_matrix.png")

    # Global GradCam analysis
    _, _, _, mots_0_25 = preprocess_before_barplot(results_one)
    mots_plus_75, _, _, _ = preprocess_before_barplot(results_two)

    #
    _, _, _, mots_0_25_wrong = preprocess_before_barplot(results_wrong_one)
    mots_plus_75_wrong, _, _, _ = preprocess_before_barplot(results_wrong_two)

    plot_barplot(mots_plus_75, path="data/08_reporting/bilstm_cnn/barplot_75.png"
                 , title="Explication globale pour la classe 1 pour les probabilités les plus élevées (>= 0.75)")
    plot_barplot(mots_0_25, path="data/08_reporting/bilstm_cnn/barplot_25.png"
                 , title="Explication globale pour la classe 0 pour les probabilités les plus faibles (<= 0.25)")
    plot_barplot(mots_plus_75_wrong, path="data/08_reporting/bilstm_cnn/barplot_75_wrong.png"
                 , title="Explication globale pour la classe 1 pour les plus fortes erreurs de prédictions")
    plot_barplot(mots_0_25_wrong, path="data/08_reporting/bilstm_cnn/barplot_25_wrong.png"
                 , title="Explication globale pour la classe 0 pour les plus fortes erreurs de prédictions")

    # COmpute bigram barplot
    plot_barplot(bigram_token_one, path="data/08_reporting/bilstm_cnn/barplot_bigram_0.png"
                 , title="Bigram pour la classe 0")
    plot_barplot(bigram_token_two, path="data/08_reporting/bilstm_cnn/barplot_bigram_1.png"
                 , title="Bigram pour la classe 1")

    pass