import io
import logging
import threading
from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image

from .config import (
    CHECKPOINT_POR_DEFECTO,
    CLASES_CIFAR10,
    DESVIACION_CIFAR10,
    MEDIA_CIFAR10,
    TAMANO_IMAGEN,
)
from .data import transformaciones_evaluacion
from .model import ResNet100, contar_capas, contar_parametros
from .utils import seleccionar_dispositivo

logger = logging.getLogger(__name__)

EntradaImagen = Image.Image | bytes | str | Path | BinaryIO


class Predictor:
    """Carga la ResNet-100 y clasifica imágenes.

    Si no existe un checkpoint entrenado, usa pesos aleatorios para que la API
    funcione igualmente; en ese caso las predicciones no tienen significado y
    ``modelo_entrenado`` es ``False``.
    """

    def __init__(self, ruta_checkpoint: str | Path | None = None, dispositivo: str = "auto"):
        self.ruta_checkpoint = Path(ruta_checkpoint or CHECKPOINT_POR_DEFECTO)
        self.dispositivo = seleccionar_dispositivo(dispositivo)
        self.modelo_entrenado = self.ruta_checkpoint.is_file()

        if self.modelo_entrenado:
            checkpoint = torch.load(self.ruta_checkpoint, map_location="cpu")
            self.clases = list(checkpoint["clases"])
            self.tamano_imagen = checkpoint.get("tamano_imagen", TAMANO_IMAGEN)
            ancho_base = checkpoint.get("ancho_base", 16)
            media = checkpoint.get("media", MEDIA_CIFAR10)
            desviacion = checkpoint.get("desviacion", DESVIACION_CIFAR10)
            self.modelo = ResNet100(num_clases=len(self.clases), ancho_base=ancho_base)
            self.modelo.load_state_dict(checkpoint["estado_modelo"])
        else:
            logger.warning(
                "No se encontró checkpoint en %s; se usan pesos aleatorios.", self.ruta_checkpoint
            )
            self.clases = list(CLASES_CIFAR10)
            self.tamano_imagen = TAMANO_IMAGEN
            media, desviacion = MEDIA_CIFAR10, DESVIACION_CIFAR10
            self.modelo = ResNet100(num_clases=len(self.clases))

        self.modelo.to(self.dispositivo).eval()
        self.transformacion = transformaciones_evaluacion(self.tamano_imagen, media, desviacion)

    @staticmethod
    def _abrir_imagen(imagen: EntradaImagen) -> Image.Image:
        if isinstance(imagen, Image.Image):
            return imagen.convert("RGB")
        if isinstance(imagen, bytes):
            imagen = io.BytesIO(imagen)
        with Image.open(imagen) as abierta:
            return abierta.convert("RGB")

    @torch.inference_mode()
    def predecir(self, imagen: EntradaImagen, top_k: int = 3) -> dict:
        tensor = self.transformacion(self._abrir_imagen(imagen)).unsqueeze(0).to(self.dispositivo)
        probabilidades = torch.softmax(self.modelo(tensor), dim=1)[0].cpu()

        k = max(1, min(top_k, len(self.clases)))
        valores, indices = probabilidades.topk(k)
        top = [
            {"clase": self.clases[i], "probabilidad": round(float(p), 6)}
            for p, i in zip(valores.tolist(), indices.tolist())
        ]
        return {
            "clase": top[0]["clase"],
            "confianza": top[0]["probabilidad"],
            "top_k": top,
            "modelo_entrenado": self.modelo_entrenado,
        }

    def informacion(self) -> dict:
        return {
            "arquitectura": "ResNet-100",
            "capas": contar_capas(self.modelo),
            "parametros": contar_parametros(self.modelo),
            "clases": self.clases,
            "tamano_imagen": self.tamano_imagen,
            "dispositivo": str(self.dispositivo),
            "modelo_entrenado": self.modelo_entrenado,
            "checkpoint": str(self.ruta_checkpoint),
        }


_instancia: Predictor | None = None
_candado = threading.Lock()


def obtener_predictor(ruta_checkpoint: str | Path | None = None) -> Predictor:
    """Devuelve un Predictor compartido (se carga una sola vez por proceso)."""
    global _instancia
    if _instancia is None:
        with _candado:
            if _instancia is None:
                _instancia = Predictor(ruta_checkpoint)
    return _instancia
