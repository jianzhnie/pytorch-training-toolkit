import copy

import torch
import torch.nn as nn


def avg_weights(
    modela: nn.Module,
    modelb: nn.Module,
    beta: float = 0.5,
    save_path: str = 'model.pth',
) -> None:
    modela_param = modela.named_parameters()
    modelb_param = modelb.named_parameters()
    dict_params = dict(modelb_param)

    for name, param_a in modela_param:
        if name in dict_params:
            dict_params[name].data.copy_(beta * param_a.data +
                                         (1 - beta) * dict_params[name].data)

    modela.load_state_dict(dict_params)
    torch.save(modela.state_dict(), save_path)
    return modela


def average_models(
    modela: nn.Module,
    modelb: nn.Module,
    beta: float = 0.5,
    save_path: str = 'model.pth',
) -> None:
    """Average the weights of two PyTorch models.

    Given two PyTorch models `modela` and `modelb`, this function computes a new
    model whose weights are the average of the weights of `modela` and `modelb`.
    The averaging is done element-wise, with a weight factor `beta` for the
    weights of `modela` and a weight factor `1-beta` for the weights of `modelb`.
    The resulting model is saved to a file at `save_path`.

    Args:
        modela (nn.Module): The first PyTorch model.
        modelb (nn.Module): The second PyTorch model.
        beta (float, optional): The weight factor for `modela`. Defaults to 0.5.
        save_path (str, optional): The path to save the resulting model. Defaults
            to "model.pth".

    Returns:
        nn.Module: The resulting PyTorch model with averaged weights.
    """
    # Create a new empty PyTorch model with the same architecture as modela and modelb
    new_model = copy.deepcopy(modela)

    new_model_state = new_model.state_dict()
    # Loop through all the parameters in modela and modelb
    for (name_a, param_a), (name_b, param_b) in zip(modela.named_parameters(),
                                                    modelb.named_parameters()):
        assert name_a == name_b
        # Compute the weighted average using beta and 1-beta
        new_param = beta * param_a + (1 - beta) * param_b
        # Assign the weighted average to the corresponding parameter in the new model
        new_model_state[param_a] = new_param

    # Save the new model to save_path
    torch.save(new_model.state_dict(), save_path)
    return new_model


if __name__ == '__main__':
    modela = nn.Transformer()
    params = modela.parameters()
    modelb = nn.Transformer()
    # avg_weights(modela, modelb)
    average_models(modela, modelb)
