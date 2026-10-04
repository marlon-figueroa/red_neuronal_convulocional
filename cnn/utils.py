import torch


def seleccionar_dispositivo(preferido: str = "auto") -> torch.device:
    if preferido != "auto":
        return torch.device(preferido)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
