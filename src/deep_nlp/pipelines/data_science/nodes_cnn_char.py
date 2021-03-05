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
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.metrics as metrics
from mlflow import log_metric, log_artifact
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
import time
import copy


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
          , cnn_num_class: int, cnn_dropout: int, cnn_path_to_save_model: str):
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

            best_model= {"model": model_clone, 'optimizer': copy.deepcopy(optimizer.state_dict())}

            # Save to pytorch to test performance models:
            save_checkpoint(model
                            , {'optimizer': optimizer.state_dict(), 'accuracy': best_eval_acc
                                , "loss": best_eval_loss, "epoch": epoch, "auc": best_eval_auroc
                               }
                            , cnn_path_to_save_model
                            )

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
             , cnn_num_threads: int, model_saved):

    corrects, avg_loss, epoch_loss, size = 0, 0, 0, 0
    predictions_all, target_all, probabilities_all= [], [], []

    test_load = DataLoader(test_data, batch_size=cnn_size_batch
                            , num_workers=cnn_num_threads
                            , drop_last=True, shuffle=True, pin_memory=True)

    # Call our model
    model= model_saved
    if cnn_cuda_allow:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model.eval()

    for batch, data in enumerate(test_load):
        input, target = data
        size += len(target)
        input = input.cuda()
        target = target.cuda()
        input = Variable(input)
        target = Variable(target)

        with torch.no_grad():
            logit = model(input)
            loss = F.nll_loss(logit, target)  # , size_average= False)

            epoch_loss += loss.item()
            predict = torch.max(logit, 1)[1].view(target.size()).data
            corrects += (predict == target.data).sum()

            predictions_all += predict.cpu().numpy().tolist()
            probabilities_all += torch.exp(logit)[:, 1].cpu().numpy().tolist()
            target_all += target.data.cpu().numpy().tolist()

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
    data_cm = metrics.confusion_matrix(target_all, predictions_all)

    positif_negatif_dict_map = {1: "positif", 0: "negatif"}

    df_cm = pd.DataFrame(data_cm, columns=[positif_negatif_dict_map[i] for i in np.unique(target_all)]
                         , index=[positif_negatif_dict_map[i] for i in np.unique(target_all)])

    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    plt.figure(figsize=(7, 6))

    sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='g')

    plt.savefig("data/08_reporting/confusion_matrix.png")
    log_artifact("data/08_reporting/confusion_matrix.png")

    plt.show()

    pass


def save_checkpoint(model, state, filename):
    # From https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/train.py
    model_is_cuda = next(model.parameters()).is_cuda
    model = model.module if model_is_cuda else model
    state['state_dict'] = model.state_dict()
    torch.save(state,filename)
    pass