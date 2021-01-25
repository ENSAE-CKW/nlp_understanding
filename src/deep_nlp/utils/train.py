import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pytorch_lightning.metrics.functional.classification import auroc

import time
from deep_nlp.utils.early_stopping import EarlyStopping
from deep_nlp.utils.utils import print_and_log, epoch_time, save_checkpoint


def train_model(model, optimizer, loss_func, params, train_load, epoch):

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
        loss = loss_func(logit, target)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params["clip"])
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


def evaluation(model, valid_load, loss_func, params):
    model.eval()

    corrects, avg_loss, epoch_loss, size= 0, 0, 0, 0
    predictions_all, target_all, probabilities, y= [], [], [], []

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
            loss = loss_func(logit, target)#, size_average= False)

            epoch_loss += loss.item()
            predict= torch.max(logit, 1)[1].view(target.size()).data
            corrects += (predict == target.data).sum()
            predictions_all.append(predict.cpu().numpy().tolist())
            target_all.append(target.data.cpu().numpy().tolist())

        # torch.cuda.synchronize()

    avg_loss = epoch_loss / len(valid_load)# avg loss (batch level)
    accuracy = 100 * corrects / size # avg acc per obs

    # # AUC
    # auroc_eval= auroc(torch.exp(logit), target)
    # print(auroc_eval)
    #
    # if params["cuda_allow"]:
    #     auroc_eval= auroc_eval.cpu().numpy()[0]
    # else:
    #     auroc_eval= auroc_eval.numpy()[0]
    auroc_eval= 0

    model.train()

    return {"loss": avg_loss, "accuracy": accuracy, "auroc": auroc_eval}


def train(params, model, optimizer, loss_func, train_data, valid_data):

    best_eval_loss= None
    best_eval_auroc= None
    best_eval_acc= None

    # Set up CUDA
    if params["cuda_allow"]:
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        model = torch.nn.DataParallel(model).cuda()

    model.train()

    # Create our log file
    if params["log_file_name"] is not None: # If we want a log train file
        f = open(params["model_path"] + params["log_file_name"], 'w+')
        f.write("Training Statement")
        f.write("------------------------------------\n")
        f.close()

    # Early stopping definition
    es= EarlyStopping(patience= params["patience"])

    # Start training
    for epoch in range(params["num_epochs"]):
        start_time = time.time()

        train_load = DataLoader(train_data, batch_size= params["size_batch"]
                                , drop_last=True,  shuffle=True, pin_memory= True)
        valid_load = DataLoader(valid_data, batch_size= params["size_batch"]
                                , drop_last= True, shuffle=True, pin_memory=True)

        model.train()
        train_loss= train_model(model, optimizer, loss_func, params, train_load, epoch)

        # Save the model if its the best encountered for so long dude (after 1 entire epoch)
        model.eval()
        valid_results = evaluation(model, valid_load, loss_func, params)

        if best_eval_loss is None or best_eval_loss > valid_results["loss"]:
            save_checkpoint(model
                            , {'optimizer': optimizer.state_dict(), 'accuracy': valid_results["accuracy"]
                                , "loss": valid_results["loss"], "epoch": epoch, "auroc": valid_results["auroc"]
                               }
                            , params["model_path"] + params["model_saved_name"]
                            )

            best_eval_loss = valid_results["loss"]
            best_eval_acc = valid_results["accuracy"]
            best_eval_auroc= valid_results["auroc"]

            if params["log_file_name"] is not None:
                print_and_log("Model Saved Epoch: {:02} | Best Val. Loss: {:.6f}(m per batch) | Acc: {:.3f}% | AUROC: "
                              "{:.6f}"  \
                              .format(epoch + 1, valid_results["loss"], valid_results["accuracy"], valid_results["auroc"])
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
                print_and_log("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
                  .format(best_eval_loss, best_eval_acc, best_eval_auroc))
            else:
                print("\nEarly stopping at epoch {:02}".format(epoch + 1))
                print("\t Best Val. encountered Loss: {:.5f} |  Accuracy: {:7.3f} | AUROC: {:.5f}\n" \
                  .format(best_eval_loss, best_eval_acc, best_eval_auroc))
            break
    pass