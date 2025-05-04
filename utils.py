import torch


def get_model_params(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)


# Set model parameters
def set_model_params(model, params):
    start = 0
    for param in model.parameters():
        end = start + param.numel()
        param.data = params[start:end].reshape(param.shape)
        start = end
