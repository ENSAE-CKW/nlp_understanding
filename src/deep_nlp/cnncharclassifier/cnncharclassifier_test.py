from deep_nlp.cnncharclassifier import charToTensor
from deep_nlp.cnncharclassifier import CNNCharClassifier
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from deep_nlp.utils.utils import *

def load(model, optimizer, params):
    checkpoint = load_checkpoint(params["model_path"] + params["model_saved_name"])

    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer


def cnn_test(model, params, test_data):
    model.eval()

    corrects, avg_loss, epoch_loss, size = 0, 0, 0, 0
    predictions_all, target_all, probabilities_all= [], [], []

    test_load = DataLoader(test_data, batch_size=params["size_batch"]
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

    fig= plt.figure(figsize= (8, 8))
    plt.plot(fpr, tpr, 'b', label= "AUC = {:.4f}".format(auroc))
    plt.legend(loc= "lower right")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.savefig(params["model_path"] + "/roc_curve.png")
    # plt.show()

    f = open(params["model_path"] + "/test_result.txt", 'w+')
    f.write("Testing results\n")
    f.write("------------------------------------\n")
    f.close()

    print_and_log("Accuracy : {:.5f} | Avg loss : {:.5f} | AUC : {:.5f}".format(accuracy, avg_loss, auroc)
                  , params["model_path"] + "/test_result.txt", print_console= True)

    pass


if __name__ == "__main__":
    params = {"lr": 0.0001, "num_epochs": 50, "size_batch": 64, "sentence_max_size": 1014
        , "num_threads": 4, "clip": 400, "verbose": False, "freq_verbose": 50
        , "cuda_allow": True, "patience": 3
        , "model_path": r"../../../data/06_models/cnn_char_classifier/allocine_classification/small_model"
        , "test_path": r"../../../data/01_raw/allocine_test.csv"
        , "valid_path": r"../../../data/01_raw/allocine_valid.csv"
        , "model_saved_name": "/small_model_allocine.pth.tar"
        }

    small_model_params = {"feature_size": 256, "kernel_one": 7
        , "kernel_two": 3, "stride_one": 1, "stride_two": 3, "seq_len": 1014
        , "output_linear": 1024, "dropout": 0.5, "num_class": 2}

    large_model_params = {"feature_size": 1024, "kernel_one": 7
        , "kernel_two": 3, "stride_one": 1, "stride_two": 3, "seq_len": 1014
        , "output_linear": 2048, "dropout": 0.5, "num_class": 2}

    test_data = charToTensor(params["test_path"], params["sentence_max_size"])

    # Adding vocabulary size to model dictionnary
    vocab_size = len(test_data.get_alphabet())
    small_model_params["feature_num"] = vocab_size

    # Call our model
    model = CNNCharClassifier(small_model_params)
    if params["cuda_allow"]:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    # Load our model Char CNN
    model, optimizer = load(model, optimizer, params)
    model.eval()  # We never know

    cnn_test(model, params, test_data)