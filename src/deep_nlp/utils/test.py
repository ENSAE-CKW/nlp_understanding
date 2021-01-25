from deep_nlp.utils.utils import load_checkpoint
from deep_nlp.cnncharclassifier import charToTensor
from deep_nlp.cnncharclassifier import CNNCharClassifier
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import auroc



def load(model, optimizer, params):
    checkpoint = load_checkpoint(params["model_path"] + params["model_saved_name"])

    model.load_state_dict(checkpoint["state_dict"])

    if params["cuda_allow"]:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model.eval()

    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer


def test(model, params, test_data):
    model.eval()

    corrects, avg_loss, epoch_loss, size = 0, 0, 0, 0
    predictions_all, target_all = [], []

    test_load = DataLoader(test_data, batch_size= params["num_batch"]
                            , shuffle=True, pin_memory=True)

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
            predictions_all.append(predict.cpu().numpy().tolist())
            target_all.append(target.data.cpu().numpy().tolist())

            # torch.cuda.synchronize()

    # avg_loss= epoch_loss/size
    # accuracy= 100*corrects/size
    avg_loss = epoch_loss / len(test_load)  # avg loss (batch level)
    accuracy = 100 * corrects / size  # avg acc per obs (size because we don't know if we dropped observations with
                                        # Dataloader

    # AUC
    auroc_test = auroc(torch.exp(logit)[:, 1], target)

    if params["cuda_allow"]:
        auroc_test = auroc_test.cpu().numpy()
    else:
        auroc_test = auroc_test.numpy()

    print(auroc_test)

    model.train()
    pass


if __name__ == "__main__":

    params= {"lr": 0.0001, "num_epochs": 50, "size_batch": 500, "sentence_max_size": 1014
        , "num_threads": 10, "clip": 400, "verbose": False, "freq_verbose": 50
        , "cuda_allow": True, "patience": 3
        , "model_path": r"../data/06_models/cnn_char_classifier/allocine_classification/small_model"
        , "test_path": r"../data/01_raw/allocine_test.csv"
        , "valid_path": r"../data/01_raw/allocine_valid.csv"
        , "model_saved_name": "/small_model_allocine.pth.tar"
        , "log_file_name": "/train_log.txt"}

    small_model_params = {"feature_size": 256, "kernel_one": 7
        , "kernel_two": 3, "stride_one": 1, "stride_two": 3, "seq_len": 1014
        , "output_linear": 1024, "dropout": 0.5, "num_class": 2}

    test_data = charToTensor(params["test_path"], params["sentence_max_size"])

    # Adding vocabulary size to model dictionnary
    vocab_size = len(test_data.get_alphabet())
    small_model_params["feature_num"] = vocab_size

    # Call our model
    model = CNNCharClassifier(small_model_params)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    # Load our model Char CNN
    model, optimizer= load(model, optimizer, params)
    model.eval() # We never know

    