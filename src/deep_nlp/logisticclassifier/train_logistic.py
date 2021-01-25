from deep_nlp.utils.early_stopping import EarlyStopping
from deep_nlp.utils.utils import print_and_log, epoch_time, save_checkpoint
from deep_nlp.logisticclassifier import LogisticClassifier
from deep_nlp.logisticclassifier.dataset_load_logistic import sentenceToTensor
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import auroc
import time



def load_checkpoint(filename):
    # Load our model
    return torch.load(filename)


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
        optimizer.step()

        # Verbose
        if batch % params["freq_verbose"] == 0:
            # ACC
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects / params["size_batch"]

            if params["log_file_name"] is not None:  # If we want a log train file
                print_and_log("Epoch[{}] Batch[{}] - loss: {:.6f}(batch)  acc: {:.3f}% ({}/{})" \
                              .format(epoch + 1, batch, loss.item(), accuracy, corrects, params["size_batch"])
                              , params["model_path"] + params["log_file_name"])

    return epoch_loss/len(train_load)


def evaluation(model, valid_data, num_batch):
    model.eval()

    corrects, avg_loss, epoch_loss, size = 0, 0, 0, 0
    predictions_all, target_all, probabilities, y = [], [], [], []

    valid_load = DataLoader(valid_data, batch_size=num_batch
                            , drop_last= True, shuffle= True)


    for batch, data in enumerate(valid_load):
        input, target = data
        size += len(target)

        if params["cuda_allow"]:
            input = input.cuda()
            target = target.cuda()

        input = Variable(input)
        target = Variable(target)


        with torch.no_grad():
            logit = model(input)
            loss = torch.nn.NLLLoss()(logit, target)#, size_average= False)

            epoch_loss += loss.item()
            predict= torch.max(logit, 1)[1].view(target.size()).data
            corrects += (predict == target.data).sum()
            predictions_all.append(predict.cpu().numpy().tolist())
            target_all.append(target.data.cpu().numpy().tolist())

        # torch.cuda.synchronize()

    # avg_loss= epoch_loss/size
    # accuracy= 100*corrects/size
    avg_loss = epoch_loss / len(valid_load)# avg loss (batch level)
    accuracy = 100 * corrects / size # avg acc per obs

    # AUC
    auroc_eval= auroc(torch.exp(logit)[:,1], target)
    print(auroc_eval)

    if params["cuda_allow"]:
        auroc_eval= auroc_eval.cpu().numpy()
    else:
        auroc_eval= auroc_eval.numpy()

    model.train()

    return {"loss": avg_loss, "accuracy": accuracy, "auroc": auroc_eval}


def train(params, continue_train= None):
    # Set up CUDA
    if params["cuda_allow"]:
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)

    # Train load
    # train_data = sentenceToTensor(params["train_path"], params["valid_path"], params["train_path"], max_bow= 50000)
    train_data = sentenceToTensor(params["train_path"], params["train_path"], params["valid_path"], max_bow= 5000)
    bow = train_data.bow()

    # Adding vocabulary size to model dictionnary
    vocab_size = train_data.len_vocab()
    num_class = params["num_class"]

    # Valid load
    valid_data = sentenceToTensor(params["train_path"], params["valid_path"], params["valid_path"], bag_of_words=bow)

    # Call our model
    model = LogisticClassifier({"feature_num": vocab_size, "num_class": num_class})
    model = torch.nn.DataParallel(model)

    if params["cuda_allow"]:
        model = torch.nn.DataParallel(model).cuda()

    model.train()

    # Optimizer definition
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])

    # If we start training from an older model
    if continue_train is None:
        checkpoint_epoch= 0

        # We reset start-batch to 0 because for the next epoch we take all the data
        best_eval_acc = 0
        best_eval_loss = None
        best_eval_auc= 0

        # Create our log file
        if params["log_file_name"] is not None: # If we want a log train file
            f = open(params["model_path"] + params["log_file_name"], 'w+')
            f.write("Training Statement CNNCharClassifier\n")
            f.write("------------------------------------\n")
            f.close()

    else:
        # Load the last model
        checkpoint = load_checkpoint(params["model_path"] + params["model_saved_name"])

        model = LogisticClassifier({"feature_num": vocab_size, "num_class": num_class})
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

        train_load = DataLoader(train_data, batch_size= params["size_batch"]
                                , drop_last=True, shuffle=True, pin_memory= True)

        model.train()
        train_loss= train_model(model, optimizer, params, train_load, epoch)

        # Save the model if its the best encountered for so long dude (after 1 entire epoch)
        model.eval()
        valid_results = evaluation(model, valid_data, num_batch=params["size_batch"])

        if best_eval_loss is None or best_eval_loss > valid_results["loss"]:
            save_checkpoint(model
                            , {'optimizer': optimizer.state_dict(), 'accuracy': valid_results["accuracy"]
                                , "loss": valid_results["loss"], "epoch": epoch
                               }
                            , params["model_path"] + params["model_saved_name"]
                            )

            best_eval_loss = valid_results["loss"]
            best_eval_acc = valid_results["accuracy"]
            best_eval_auc= valid_results["auroc"]

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
                          .format(best_eval_loss, best_eval_acc, best_eval_auc)
                          , params["model_path"] + params["log_file_name"])
        else:
            print("\nEpoch: {:02} | Time: {}m {}s".format(epoch + 1, epoch_min, epoch_sec))
            print("\tTrain Loss: {:.5f}".format(train_loss))
            print("\t Val. Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
                  .format(valid_results["loss"], valid_results["accuracy"], valid_results["auroc"]))
            print("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
                  .format(best_eval_loss, best_eval_acc, best_eval_auc))

        # Early stopping callback
        if es.step(valid_results["loss"]):
            if params["log_file_name"] is not None:
                print_and_log("\nEarly stopping at epoch {:02}".format(epoch + 1)
                              , params["model_path"] + params["log_file_name"])
                print_and_log("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
                              .format(best_eval_loss, best_eval_acc, best_eval_auc)
                              , params["model_path"] + params["log_file_name"])
            else:
                print("\nEarly stopping at epoch {:02}".format(epoch + 1))
                print("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
                      .format(best_eval_loss, best_eval_acc, best_eval_auc))
            break

    pass


if __name__ == "__main__":

    params = {"lr": 0.02, "num_epochs": 50, "num_class": 2, "size_batch": 64
        , "verbose": False, "freq_verbose": 500, "cuda_allow": True, "patience": 5
        , "model_path": r"../data/06_models/logisticclassifier/allocine_classification"
        , "train_path": r"../data/01_raw/allocine_train.csv"
        , "valid_path": r"../data/01_raw/allocine_valid.csv"
        , "model_saved_name": "/model_allocine.pth.tar"
        , "log_file_name": "/train_log.txt"}

    # # Train load
    # # train_data = sentenceToTensor(params["train_path"], params["valid_path"], params["train_path"], max_bow= 50000)
    # train_data = sentenceToTensor(params["train_path"], params["train_path"], params["valid_path"], max_bow= 50000)
    # bow = train_data.bow()
    #
    # # print(bow)
    # # print(len(bow))
    #
    # # Adding vocabulary size to model dictionnary
    # vocab_size = train_data.len_vocab()
    # num_class = params["num_class"]
    #
    # # Valid load
    # valid_data = sentenceToTensor(params["train_path"], params["valid_path"], params["valid_path"], bag_of_words=bow)
    #
    # # Call our model
    # model = LogisticClassifier({"feature_num": vocab_size, "num_class": num_class})
    # model = torch.nn.DataParallel(model)
    #
    # # Optimizer definition
    # optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])
    #
    # # Loss function definition
    # loss_func= torch.nn.NLLLoss()

    train(params) # change path into model