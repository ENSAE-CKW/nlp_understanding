# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from deep_nlp.cnncharclassifier import CNNCharClassifier
from deep_nlp.utils.early_stopping import EarlyStopping
from deep_nlp.utils.utils import *
from src.deep_nlp.grad_cam.plot import plot_cm, plot_barplot
from deep_nlp.grad_cam.utils.letter import rebuild_text, prepare_heatmap, LetterToToken
from src.deep_nlp.grad_cam.utils.token import order_tokens_by_importance, find_ngram
from src.deep_nlp.grad_cam.utils import preprocess_before_barplot

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.metrics as metrics
from mlflow import log_metric
import seaborn as sns
sns.set()
import numpy as np
import time

def train_model(model, optimizer, train_load, epoch
                , cnn_cuda_allow, cnn_clip, cnn_freq_verbose, cnn_size_batch):

    model.train()

    epoch_loss = 0

    for batch, data in enumerate(train_load):
        input, target = data

        if cnn_cuda_allow:
            input = input.cuda()
            target = target.cuda()

        input = Variable(input)
        target = Variable(target)
        logit = model(input)
        loss = F.nll_loss(logit, target)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cnn_clip)
        optimizer.step()

        # Verbose
        if batch % cnn_freq_verbose == 0:
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects / int(cnn_size_batch)

            print("Epoch[{}] Batch[{}] - loss: {:.6f}(sum per batch)  acc: {:.3f}% ({}/{})" \
                          .format(epoch + 1, batch, loss.item(), accuracy, corrects, cnn_size_batch))


    return epoch_loss/len(train_load)


def evaluation(model, valid_data
               , cnn_size_batch, cnn_num_threads):
    model.eval()

    corrects, avg_loss, epoch_loss, size= 0, 0, 0, 0
    predictions_all, target_all, probabilities_all= [], [], []

    valid_load = DataLoader(valid_data, batch_size= cnn_size_batch
                            , num_workers= cnn_num_threads
                            , drop_last= True, shuffle= True, pin_memory= True)

    for batch, data in enumerate(valid_load):
        input, target = data
        size += len(target)
        input = input.cuda()
        target = target.cuda()
        input = Variable(input)
        target = Variable(target)

        with torch.no_grad():
            logit = model(input)
            loss = F.nll_loss(logit, target)#, size_average= False)

            epoch_loss += loss.item()
            predict= torch.max(logit, 1)[1].view(target.size()).data
            corrects += (predict == target.data).sum()

            predictions_all += predict.cpu().numpy().tolist()
            probabilities_all += torch.exp(logit)[:, 1].cpu().numpy().tolist()
            target_all += target.data.cpu().numpy().tolist()

    avg_loss = epoch_loss / len(valid_load) # avg loss (batch level)
    accuracy = 100 * corrects / size # avg acc per obs

    ## AUC and ROC Curve
    fpr, tpr, threshold= metrics.roc_curve(target_all, probabilities_all)
    auroc= metrics.auc(fpr, tpr)

    # print(accuracy.cpu().numpy().tolist())
    # print(avg_loss)
    # print(auroc)

    model.train()

    return {"loss": avg_loss, "accuracy": accuracy.cpu().numpy().tolist(), "auroc": auroc}


def train(train_data, valid_data, cnn_freq_verbose: int, cnn_clip: int
          , cnn_cuda_allow: bool, cnn_feature_num: int, cnn_lr: float, cnn_patience: int, cnn_num_epochs: int
          , cnn_size_batch: int, cnn_num_threads: int, cnn_sequence_len: int, cnn_feature_size: int
          , cnn_kernel_one: int, cnn_kernel_two: int, cnn_stride_one: int, cnn_stride_two: int, cnn_output_linear: int
          , cnn_num_class: int, cnn_dropout: int):
    # Set up CUDA
    if cnn_cuda_allow:
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)

    # Call our model
    kwargs= {"sequence_len": cnn_sequence_len, "feature_num": cnn_feature_num
         , "feature_size": cnn_feature_size, "kernel_one": cnn_kernel_one
         , "kernel_two": cnn_kernel_two, "stride_one": cnn_stride_one
         , "stride_two": cnn_stride_two, "output_linear": cnn_output_linear
         , "num_class": cnn_num_class, "dropout": cnn_dropout}

    model= CNNCharClassifier(**kwargs)
    model= torch.nn.DataParallel(model)

    if cnn_cuda_allow:
        model = torch.nn.DataParallel(model).cuda()

    model.train()

    optimizer= optim.Adam(model.parameters(), lr= cnn_lr)

    # We reset start-batch to 0 because for the next epoch we take all the data
    best_eval_acc= 0
    best_eval_loss= None
    best_eval_auroc= 0
    best_model= {}

    # Early stopping definition
    es= EarlyStopping(patience= cnn_patience)

    # Start training
    for epoch in range(cnn_num_epochs):
        start_time = time.time()

        train_load = DataLoader(train_data, batch_size= cnn_size_batch
                                , num_workers= cnn_num_threads
                                , drop_last=True, shuffle=True, pin_memory= True)

        model.train()
        train_loss= train_model(model= model, optimizer= optimizer, train_load= train_load, epoch= epoch
                                , cnn_cuda_allow= cnn_cuda_allow, cnn_clip= cnn_clip
                                , cnn_freq_verbose= cnn_freq_verbose, cnn_size_batch=cnn_size_batch)


        # Save the model if its the best encountered for so long dude (after 1 entire epoch)
        model.eval()
        valid_results = evaluation(model= model, valid_data= valid_data,  cnn_size_batch= cnn_size_batch
                            ,cnn_num_threads= cnn_num_threads)

        if best_eval_loss is None or best_eval_loss > valid_results["loss"]:

            # Instance a new model, copy the parameters of the best model and then save it
            model_clone = CNNCharClassifier(**kwargs)
            model_clone = torch.nn.DataParallel(model_clone)
            if cnn_cuda_allow:
                model_clone = torch.nn.DataParallel(model_clone).cuda()

            model_clone.load_state_dict(model.state_dict())

            best_model= {"model": model_clone.state_dict()}


            best_eval_loss = valid_results["loss"]
            best_eval_acc = valid_results["accuracy"]
            best_eval_auroc= valid_results["auroc"]

            print("Model Saved Epoch: {:02} | Best Val. Loss: {:.6f}(m per batch) | Acc: {:.3f}% | AUROC: "
                          "{:.6f}" \
                          .format(epoch + 1, valid_results["loss"], valid_results["accuracy"],
                                  valid_results["auroc"]))

        # Compute time and print to logfile
        end_time = time.time()
        epoch_min, epoch_sec = epoch_time(start_time, end_time)

        print("\nEpoch: {:02} | Time: {}m {}s".format(epoch + 1, epoch_min, epoch_sec))
        print("\tTrain Loss: {:.5f}".format(train_loss))
        print("\t Val. Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
              .format(valid_results["loss"], valid_results["accuracy"], valid_results["auroc"]))
        print("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
              .format(best_eval_loss, best_eval_acc, best_eval_auroc))

        # MLflow log metrics per epoch
        log_metric(key="Train_loss", value= train_loss, step=epoch + 1)
        log_metric(key="Validation_loss", value= valid_results["loss"], step=epoch + 1)
        log_metric(key="Accuracy", value= valid_results["accuracy"], step=epoch + 1)
        log_metric(key= "AUC", value= valid_results["auroc"], step= epoch+1)

        # Early stopping callback
        if es.step(valid_results["loss"]):
            print("\nEarly stopping at epoch {:02}".format(epoch + 1))
            print("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f}\n" \
                  .format(best_eval_loss, best_eval_acc))

            log_metric(key="Best_validation_loss", value=best_eval_loss, step=epoch + 1)
            log_metric(key="Best_AUC", value=best_eval_auroc, step=epoch + 1)
            log_metric(key="Best_Accuracy", value=best_eval_acc, step=epoch + 1)

            break

    model.eval()
    return best_model["model"]


def cnn_test(test_data, cnn_cuda_allow: bool, cnn_size_batch: int
             , cnn_num_threads: int, model_saved, cnn_feature_num: int, cnn_sequence_len: int
             , cnn_feature_size: int, cnn_kernel_one: int, cnn_kernel_two: int, cnn_stride_one: int
             , cnn_stride_two: int, cnn_output_linear: int, cnn_num_class: int, cnn_dropout: int
             , type_map: str, seuil: float, type_agg: str):

    corrects, avg_loss, epoch_loss, size = 0, 0, 0, 0
    predictions_all, target_all, probabilities_all= [], [], []
    results_two, results_one= [], []
    results_wrong_two, results_wrong_one= [], []
    bigram_token_two, bigram_token_one= [], []


    test_load = DataLoader(test_data, batch_size= 1
                            , num_workers=cnn_num_threads)

    alphabet = test_data.get_alphabet() + " "

    # Call our model
    model_parameters = {"sequence_len": cnn_sequence_len, "feature_num": cnn_feature_num
        , "feature_size": cnn_feature_size, "kernel_one": cnn_kernel_one
        , "kernel_two": cnn_kernel_two, "stride_one": cnn_stride_one
        , "stride_two": cnn_stride_two, "output_linear": cnn_output_linear
        , "num_class": cnn_num_class, "dropout": cnn_dropout}

    model = CNNCharClassifier(**model_parameters)

    if cnn_cuda_allow:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(model_saved)

    state_dict = model.module.state_dict()  # delete module to allow cpu loading

    cpu_model = CNNCharClassifier(**model_parameters).cpu()
    cpu_model.load_state_dict(state_dict)

    cpu_model.eval()
    del model # Memory my boy is not unlimited

    i = 0
    for batch, data in enumerate(test_load):
        input, target = data
        size += len(target)

        logit = cpu_model(input)
        loss = F.nll_loss(logit, target)

        epoch_loss += loss.item()
        predict = torch.max(logit, 1)[1].view(target.size()).data
        corrects += (predict == target.data).sum()

        predictions_all += predict.detach().cpu().numpy().tolist()
        probabilities_all += torch.exp(logit)[:, 1].detach().cpu().numpy().tolist()
        target_all += target.data.detach().cpu().numpy().tolist()

        # Compute GradCam for gloabal interpretability
        # Proba for class 1
        proba_1 = torch.exp(logit)[:, 1].detach().cpu().numpy()[0]
        class_explanation = 0
        other_class_explanation = 1 if class_explanation == 0 else 0
        difference_classification = target - proba_1


        # Rebuild text
        rebuild_sentence = rebuild_text(text= input
                                        , alphabet=alphabet
                                        , space_index= len(alphabet) - 1
                                        , sequence_len=cnn_sequence_len)

        if proba_1 >= 0.5: # The model predict 1. We want to understand why. So we compute GradCam for the class 1
            explanations_class_two = cpu_model.get_heatmap(text= input
                                                       , num_class=other_class_explanation
                                                       , dim=[0, 2]
                                                       , type_map=type_map)[-1]

            heatmap_token_level, tokens = compute_heatmap_token_level(heatmap=explanations_class_two
                                                                      , sentence= rebuild_sentence
                                                                      , type_agg=type_agg)

            # Clean list and heatmap to take off "" element in tokens (only if cleaned_tokens)
            tokens_good_index = np.where(tokens != "")[0]
            if tokens.shape[0] > 1:  # if composed of more than 1 token
                # print(1)
                # print(tokens_good_index)
                tokens = tokens[tokens_good_index]
                heatmap_token_level = heatmap_token_level[tokens_good_index]
            else:
                continue
            # Find bigram pairwise index
            best_word_explanation_index_two = np.where(heatmap_token_level >= seuil)[0]
            bigram_index = find_ngram(best_word_explanation_index_two, occurence=2)

            # Generate all important bigram
            bigram_token_two += [" ".join(t) for t in
                                 [tokens[i].tolist() for i in bigram_index]
                                 ]

            best_word_explanation_two = order_tokens_by_importance(heatmap= heatmap_token_level
                                                                   , tokens= tokens
                                                                   , threshold= seuil)

            explications_pour_plot_two = {"mots_expli": best_word_explanation_two
                , "prob": proba_1}

            results_two.append([explications_pour_plot_two, target])

            # If we did a big mistake (here means we predict 1 with a proba near of 1, but in fact, the true
            # label was 0
            # So we save the result to understand why the model made a such mistake
            # The idea itsto look at the grad cam for the class the model thought the sentence was
            # Here, we save the gradcam for the class 1 because the model was really sure about is classification
            if np.abs(difference_classification) >= seuil:
                results_wrong_two.append(results_two[-1])

        else: # The model classified it as a 0 (negative review)

            explanations_class_one = cpu_model.get_heatmap(text=input
                                                           , num_class=class_explanation
                                                           , dim=[0, 2]
                                                           , type_map=type_map)[-1]

            heatmap_token_level, tokens= compute_heatmap_token_level(heatmap= explanations_class_one
                                                                     , sentence= rebuild_sentence
                                                                     , type_agg= type_agg)

            # Clean list and heatmap to take off "" element in tokens (only if cleaned_tokens)
            tokens_good_index = np.where(tokens != "")[0]

            if tokens.shape[0] > 1:  # if composed of more than 1 token
                # print(0)
                # print(tokens_good_index)
                tokens = tokens[tokens_good_index]
                heatmap_token_level = heatmap_token_level[tokens_good_index]
            else: # If cleaned tokens, ignore it
                continue

            # Find bigram pairwise index
            best_word_explanation_index_one = np.where(heatmap_token_level >= seuil)[0]
            bigram_index = find_ngram(best_word_explanation_index_one, occurence=2)
            # Generate all important bigram
            bigram_token_one += [" ".join(t) for t in
                                 [tokens[i].tolist() for i in bigram_index]
                                 ]

            best_word_explanation_one = order_tokens_by_importance(heatmap=heatmap_token_level
                                                                   , tokens=tokens
                                                                   , threshold=seuil)

            explications_pour_plot_one = {"mots_expli": best_word_explanation_one
                , "prob": proba_1}

            results_one.append([explications_pour_plot_one, target])

            # If we did a big mistake (here means we predict 1 with a proba near of 1, but in fact, the true
            # label was 0
            if np.abs(difference_classification) >= seuil:
                results_wrong_one.append(results_one[-1])

        i += 1
        if i % 1000 == 0:
            print(i)

    avg_loss = epoch_loss / len(test_load)  # avg loss (batch level)
    accuracy = 100 * corrects / size  # avg acc per obs

    ## AUC and ROC Curve
    fpr, tpr, threshold = metrics.roc_curve(target_all, probabilities_all)
    auroc = metrics.auc(fpr, tpr)


    log_metric(key="Test_loss", value= avg_loss)
    log_metric(key="Accuracy_test", value= accuracy.cpu().numpy().tolist())
    log_metric(key="AUC_test", value= auroc)

    print("Accuracy : {:.5f} | Avg loss : {:.5f} | AUC : {:.5f}".format(accuracy.cpu().numpy().tolist()
                                                                        , avg_loss, auroc))

    # Confusion matrix
    plot_cm(target_all, predictions_all, path="data/08_reporting/cnn_char/confusion_matrix.png")

    # PLot global gradcam interpretability
    # Global GradCam analysis
    _, _, _, mots_0_25 = preprocess_before_barplot(results_one)
    mots_plus_75, _, _, _ = preprocess_before_barplot(results_two)

    #
    _, _, _, mots_0_25_wrong = preprocess_before_barplot(results_wrong_one)
    mots_plus_75_wrong, _, _, _ = preprocess_before_barplot(results_wrong_two)

    plot_barplot(mots_plus_75, path="data/08_reporting/cnn_char/barplot_75.png"
                 , title="Explication globale pour la classe 1 pour les plus grosses probabilités (>= 0.75)")
    plot_barplot(mots_0_25, path="data/08_reporting/cnn_char/barplot_25.png"
                 , title="Explication globale pour la classe 0 pour les plus faibles probabilités (<= 0.25)")
    plot_barplot(mots_plus_75_wrong, path="data/08_reporting/cnn_char/barplot_75_wrong.png"
                 , title="Explication globale pour la classe 1 pour les plus grosses erreurs de prédictions")
    plot_barplot(mots_0_25_wrong, path="data/08_reporting/cnn_char/barplot_25_wrong.png"
                 , title="Explication globale pour la classe 0 pour les plus grosses erreurs de prédictions")

    # COmpute bigram barplot
    plot_barplot(bigram_token_one, path="data/08_reporting/cnn_char/barplot_bigram_0.png"
                 , title="Bigram pour la classe 0")
    plot_barplot(bigram_token_two, path="data/08_reporting/cnn_char/barplot_bigram_1.png"
                 , title="Bigram pour la classe 1")
pass


def compute_heatmap_token_level(heatmap, sentence, type_agg):
    # Resize heatmap Brutal method
    heatmap_match_sentence_size_invert = prepare_heatmap(heatmap=heatmap
                                                         , text=sentence)

    # Transform character level to token one
    letter_to_token = LetterToToken(text=sentence
                                    , heatmap=heatmap_match_sentence_size_invert)

    results_dict = letter_to_token.transform_letter_to_token(type=type_agg)
    tokens = np.array(results_dict["cleaned_tokens"])
    heatmap_token_level = results_dict["heatmap"]

    return heatmap_token_level, tokens