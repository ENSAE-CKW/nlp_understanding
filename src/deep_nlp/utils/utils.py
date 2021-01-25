import torch

def print_and_log(sentence: str, filename: str, print_console= True):
    # We should perform a try/except to insure the filename exists
    f = open(filename, 'a')
    print(sentence, file= f)
    f.close()
    if print_console:
        print(sentence)
    pass


def epoch_time(start_time, end_time):
    # From https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(model, state, filename):
    # From https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/train.py
    model_is_cuda = next(model.parameters()).is_cuda
    model = model.module if model_is_cuda else model
    state['state_dict'] = model.state_dict()
    torch.save(state,filename)
    pass


def load_checkpoint(filename):
    # Load our model
    return torch.load(filename)


def load(model, optimizer, params):
    checkpoint = load_checkpoint(params["model_path"] + params["model_saved_name"])

    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer