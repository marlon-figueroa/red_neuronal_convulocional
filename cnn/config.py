from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ_PROYECTO / "data"
DIR_CHECKPOINTS = RAIZ_PROYECTO / "checkpoints"
CHECKPOINT_POR_DEFECTO = DIR_CHECKPOINTS / "resnet100.pt"

TAMANO_IMAGEN = 32

CLASES_CIFAR10 = [
    "avion",
    "automovil",
    "pajaro",
    "gato",
    "venado",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camion",
]

MEDIA_CIFAR10 = (0.4914, 0.4822, 0.4465)
DESVIACION_CIFAR10 = (0.2470, 0.2435, 0.2616)
