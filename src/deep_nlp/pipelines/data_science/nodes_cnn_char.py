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

    print(accuracy)
    print(avg_loss)
    print(auroc)

    model.train()

    return {"loss": avg_loss, "accuracy": accuracy, "auroc": auroc}


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
            # save_checkpoint(model
            #                 , {'optimizer': optimizer.state_dict(), 'accuracy': valid_results["accuracy"]
            #                     , "loss": valid_results["loss"], "epoch": epoch
            #                    }
            #                 , parameters["cnn_model_path"] + parameters["cnn_model_saved_name"]
            #                 )
            best_model= {"model": model, 'optimizer': optimizer.state_dict()}

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

        # Early stopping callback
        if es.step(valid_results["loss"]):
            print("\nEarly stopping at epoch {:02}".format(epoch + 1))
            print("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f}\n" \
                  .format(best_eval_loss, best_eval_acc))
            break

    model.eval()
    return best_model["model"]


def cnn_test(model, test_data, parameters):
    model.eval()

    corrects, avg_loss, epoch_loss, size = 0, 0, 0, 0
    predictions_all, target_all, probabilities_all= [], [], []

    test_load = DataLoader(test_data, batch_size=parameters["cnn_size_batch"]
                           , pin_memory=True)

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
    accuracy = 100 * corrects / size  # avg acc per obs (size because we don't know if we dropped observations with
                                        # Dataloader

    # AUC and ROC Curve
    fpr, tpr, threshold= metrics.roc_curve(target_all, probabilities_all)
    auroc= metrics.auc(fpr, tpr)

    print(accuracy)
    print(avg_loss)
    print(auroc)

    print("Accuracy : {:.5f} | Avg loss : {:.5f} | AUC : {:.5f}".format(accuracy, avg_loss, auroc))

    pass
