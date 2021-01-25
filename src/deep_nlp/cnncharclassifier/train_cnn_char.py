from deep_nlp.cnncharclassifier import charToTensor
from deep_nlp.cnncharclassifier import CNNCharClassifier
from deep_nlp.utils.early_stopping import EarlyStopping
from deep_nlp.utils.utils import *
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import auroc
import time



def train_model(model, optimizer, params, train_load, epoch):

    model.train()

    epoch_loss = 0

    for batch, data in enumerate(train_load):
        input, target = data

        if params["cuda_allow"]:
            input = input.cuda()
            target = target.cuda()

        input = Variable(input)
        target = Variable(target)

        logit = model(input)
        loss = F.nll_loss(logit, target)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params["clip"])
        optimizer.step()

        # Verbose
        if batch % params["freq_verbose"] == 0:
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects / params["size_batch"]

            if params["log_file_name"] is not None:  # If we want a log train file
                print_and_log("Epoch[{}] Batch[{}] - loss: {:.6f}(sum per batch)  acc: {:.3f}% ({}/{})" \
                              .format(epoch + 1, batch, loss.item(), accuracy, corrects, params["size_batch"])
                              , params["model_path"] + params["log_file_name"])

    return epoch_loss/len(train_load)


def evaluation(model, valid_data, num_batch, num_threads):
    model.eval()

    corrects, avg_loss, epoch_loss, size= 0, 0, 0, 0
    predictions_all, target_all= [], []

    valid_load = DataLoader(valid_data, batch_size=num_batch, num_workers=num_threads
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
            predictions_all.append(predict.cpu().numpy().tolist())
            target_all.append(target.data.cpu().numpy().tolist())

            # torch.cuda.synchronize()

    # avg_loss= epoch_loss/size
    # accuracy= 100*corrects/size
    avg_loss = epoch_loss / len(valid_load) # avg loss (batch level)
    accuracy = 100 * corrects / size # avg acc per obs

    # AUC
    auroc_eval= auroc(torch.exp(logit)[:,1], target)

    if params["cuda_allow"]:
        auroc_eval= auroc_eval.cpu().numpy()
    else:
        auroc_eval= auroc_eval.numpy()

    print(auroc_eval)

    model.train()

    return {"loss": avg_loss, "accuracy": accuracy, "auroc": auroc_eval}


def train(model_params, params, continue_train= None):
    # Set up CUDA
    if params["cuda_allow"]:
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)

    # Train load
    train_data = charToTensor(params["train_path"], params["sentence_max_size"])

    # Adding vocabulary size to model dictionnary
    vocab_size = len(train_data.get_alphabet())
    model_params["feature_num"] = vocab_size

    # Valid load
    valid_data = charToTensor(params["valid_path"], params["sentence_max_size"])

    # Call our model
    model= CNNCharClassifier(model_params)
    model= torch.nn.DataParallel(model)

    if params["cuda_allow"]:
        model = torch.nn.DataParallel(model).cuda()

    model.train()

    optimizer= optim.Adam(model.parameters(), lr= params["lr"])

    # If we start training from an older model
    if continue_train is None:
        checkpoint_epoch= 0

        # We reset start-batch to 0 because for the next epoch we take all the data
        best_eval_acc = 0
        best_eval_loss = None
        best_eval_auroc= 0

        # Create our log file
        if params["log_file_name"] is not None: # If we want a log train file
            f = open(params["model_path"] + params["log_file_name"], 'w+')
            f.write("Training Statement CNNCharClassifier\n")
            f.write("------------------------------------\n")
            f.close()

    else:
        # Load the last model
        checkpoint = load_checkpoint(params["model_path"] + params["model_saved_name"])

        model = CNNCharClassifier(model_params)
        model.load_state_dict(checkpoint["state_dict"])
        model = torch.nn.DataParallel(model)

        if params["cuda_allow"]:
            model = torch.nn.DataParallel(model).cuda()

        model.train()

        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # We stopped the model at the end of an epoch (hypothesis)
        checkpoint_epoch= checkpoint["epoch"] + 1

        best_eval_acc = checkpoint["accuracy"]
        best_eval_loss = checkpoint["loss"]

        if params["log_file_name"] is not None: # If we want a log train file
            try:
                f= open(params["model_path"] + params["log_file_name"], 'a')
                f.write("Training Resuming\n")
                f.write("------------------------------------\n")
                f.close()
            except:
                # Create our log file
                f = open(params["model_path"] + params["log_file_name"], 'w+')
                f.write("Training Statement CNNCharClassifier (resuming but without log file)\n")
                f.write("------------------------------------\n")
                f.close()

    # Early stopping definition
    es= EarlyStopping(patience= params["patience"])

    # Start training
    for epoch in range(checkpoint_epoch, params["num_epochs"] - checkpoint_epoch):
        start_time = time.time()

        train_load = DataLoader(train_data, batch_size= params["size_batch"], num_workers=params["num_threads"]
                                , drop_last=True, shuffle=True, pin_memory= True)

        model.train()
        train_loss= train_model(model, optimizer, params, train_load, epoch)

        # Save the model if its the best encountered for so long dude (after 1 entire epoch)
        model.eval()
        valid_results = evaluation(model, valid_data, num_batch=params["size_batch"]
                                   , num_threads=params["num_threads"])

        if best_eval_loss is None or best_eval_loss > valid_results["loss"]:
            save_checkpoint(model
                            , {'optimizer': optimizer.state_dict(), 'accuracy': valid_results["accuracy"]
                                , "loss": valid_results["loss"], "epoch": epoch
                               }
                            , params["model_path"] + params["model_saved_name"]
                            )

            best_eval_loss = valid_results["loss"]
            best_eval_acc = valid_results["accuracy"]
            best_eval_auroc= valid_results["auroc"]

            if params["log_file_name"] is not None:
                print_and_log("Model Saved Epoch: {:02} | Best Val. Loss: {:.6f}(m per batch) | Acc: {:.3f}% | AUROC: "
                              "{:.6f}" \
                              .format(epoch + 1, valid_results["loss"], valid_results["accuracy"],
                                      valid_results["auroc"])
                              , params["model_path"] + params["log_file_name"])

        # Compute time and print to logfile
        end_time = time.time()
        epoch_min, epoch_sec = epoch_time(start_time, end_time)

        if params["log_file_name"] is not None:
            print_and_log("\nEpoch: {:02} | Time: {}m {}s".format(epoch + 1, epoch_min, epoch_sec)
                          , params["model_path"] + params["log_file_name"])
            print_and_log("\tTrain Loss: {:.5f}".format(train_loss)
                          , params["model_path"] + params["log_file_name"])
            print_and_log("\t Val. Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
                          .format(valid_results["loss"], valid_results["accuracy"], valid_results["auroc"])
                          , params["model_path"] + params["log_file_name"])
            print_and_log("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
                          .format(best_eval_loss, best_eval_acc, best_eval_auroc)
                          , params["model_path"] + params["log_file_name"])
        else:
            print("\nEpoch: {:02} | Time: {}m {}s".format(epoch + 1, epoch_min, epoch_sec))
            print("\tTrain Loss: {:.5f}".format(train_loss))
            print("\t Val. Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
                  .format(valid_results["loss"], valid_results["accuracy"], valid_results["auroc"]))
            print("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
                  .format(best_eval_loss, best_eval_acc, best_eval_auroc))

        # Early stopping callback
        if es.step(valid_results["loss"]):
            if params["log_file_name"] is not None:
                print_and_log("\nEarly stopping at epoch {:02}".format(epoch + 1)
                              , params["model_path"] + params["log_file_name"])
                print_and_log("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f}\n" \
                              .format(best_eval_loss, best_eval_acc)
                              , params["model_path"] + params["log_file_name"])
            else:
                print("\nEarly stopping at epoch {:02}".format(epoch + 1))
                print("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f}\n" \
                      .format(best_eval_loss, best_eval_acc))
            break

    pass


if __name__ == "__main__":

    params= {"lr": 0.0001, "num_epochs": 50, "size_batch": 100, "sentence_max_size": 1014
        , "num_threads": 4, "clip": 400, "verbose": False, "freq_verbose": 100
        , "cuda_allow": True, "patience": 3
        , "model_path": r"../data/06_models/cnn_char_classifier/allocine_classification/large_model"
        , "train_path": r"../data/01_raw/allocine_train.csv"
        , "valid_path": r"../data/01_raw/allocine_valid.csv"
        , "model_saved_name": "/large_model_allocine.pth.tar"
        , "log_file_name": "/train_log.txt"}

    small_model_params= {"feature_size": 256, "kernel_one": 7
    , "kernel_two": 3, "stride_one": 1, "stride_two": 3, "seq_len": 1014
    , "output_linear": 1024, "dropout": 0.5, "num_class": 2}

    large_model_params= {"feature_size": 1024, "kernel_one": 7
    , "kernel_two": 3, "stride_one": 1, "stride_two": 3, "seq_len": 1014
    , "output_linear": 2048, "dropout": 0.5, "num_class": 2}


    train(large_model_params, params, continue_train= None)
