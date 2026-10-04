from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import CLASES_CIFAR10, DESVIACION_CIFAR10, DIR_DATOS, MEDIA_CIFAR10, TAMANO_IMAGEN


def transformaciones_entrenamiento(tamano: int, media, desviacion) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(tamano),
            transforms.RandomCrop(tamano, padding=tamano // 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(media, desviacion),
        ]
    )


def transformaciones_evaluacion(tamano: int, media, desviacion) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(tamano),
            transforms.CenterCrop(tamano),
            transforms.ToTensor(),
            transforms.Normalize(media, desviacion),
        ]
    )


def crear_dataloaders(
    dataset: str = "cifar10",
    ruta_datos: Path | None = None,
    tamano_imagen: int = TAMANO_IMAGEN,
    tamano_lote: int = 128,
    trabajadores: int = 2,
):
    """Devuelve (loader_entrenamiento, loader_validacion, clases).

    - dataset="cifar10": descarga CIFAR-10 automáticamente en ``ruta_datos``.
    - dataset="carpeta": espera ``ruta_datos/train/<clase>/*.jpg`` y
      ``ruta_datos/val/<clase>/*.jpg`` (formato ImageFolder).
    """
    media, desviacion = MEDIA_CIFAR10, DESVIACION_CIFAR10
    t_entrenamiento = transformaciones_entrenamiento(tamano_imagen, media, desviacion)
    t_evaluacion = transformaciones_evaluacion(tamano_imagen, media, desviacion)

    if dataset == "cifar10":
        raiz = Path(ruta_datos or DIR_DATOS)
        entrenamiento = datasets.CIFAR10(raiz, train=True, download=True, transform=t_entrenamiento)
        validacion = datasets.CIFAR10(raiz, train=False, download=True, transform=t_evaluacion)
        clases = list(CLASES_CIFAR10)
    elif dataset == "carpeta":
        if ruta_datos is None:
            raise ValueError("Con dataset='carpeta' debes indicar ruta_datos")
        raiz = Path(ruta_datos)
        entrenamiento = datasets.ImageFolder(raiz / "train", transform=t_entrenamiento)
        validacion = datasets.ImageFolder(raiz / "val", transform=t_evaluacion)
        clases = entrenamiento.classes
    else:
        raise ValueError(f"Dataset no soportado: {dataset}")

    loader_entrenamiento = DataLoader(
        entrenamiento, batch_size=tamano_lote, shuffle=True, num_workers=trabajadores
    )
    loader_validacion = DataLoader(
        validacion, batch_size=tamano_lote, shuffle=False, num_workers=trabajadores
    )
    return loader_entrenamiento, loader_validacion, clases
