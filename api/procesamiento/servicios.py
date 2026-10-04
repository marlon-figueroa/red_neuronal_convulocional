from django.conf import settings

from cnn.predictor import obtener_predictor


def predictor():
    return obtener_predictor(settings.CNN_CHECKPOINT)


def clasificar(archivo, top_k: int = 3) -> dict:
    """Clasifica un archivo subido y deja el puntero al inicio para poder guardarlo."""
    archivo.seek(0)
    resultado = predictor().predecir(archivo, top_k=top_k)
    archivo.seek(0)
    return resultado
