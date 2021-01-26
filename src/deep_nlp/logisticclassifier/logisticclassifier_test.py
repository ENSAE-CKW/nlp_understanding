from deep_nlp.logisticclassifier import LogisticClassifier
from deep_nlp.logisticclassifier.dataset_load_logistic import sentenceToTensor
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from deep_nlp.utils.utils import *


def logisticclassi_test(model, params, test_data):
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
    params = {"lr": 0.001, "num_epochs": 50, "num_class": 2, "size_batch": 86
        , "verbose": False, "freq_verbose": 86, "cuda_allow": True, "patience": 5
        , "model_path": r"../../../data/06_models/logisticclassifier/allocine_classification"
        , "train_path": r"../../../data/01_raw/allocine_train.csv"
        , "valid_path": r"../../../data/01_raw/allocine_valid.csv"
        , "test_path": r"../../../data/01_raw/allocine_test.csv"
        , "model_saved_name": "/model_allocine.pth.tar"
        , "log_file_name": "/train_log.txt"}

    stt_params = {"train_path": params["train_path"]
        , "valid_path": params["valid_path"]
        , "max_features": 100
        , "vocabulary": None}

    test_data = sentenceToTensor(params=stt_params
                                  , data_path=params["test_path"])


    # Call our model
    model = LogisticClassifier({"feature_num": stt_params["max_features"], "num_class": params["num_class"]})
    if params["cuda_allow"]:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    # Load our model Char CNN
    model, optimizer = load(model, optimizer, params)
    model.eval()  # We never know

    logisticclassi_test(model, params, test_data)