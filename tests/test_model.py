import io

import pytest
import torch
from PIL import Image

from cnn.model import ResNet100, contar_capas
from cnn.predictor import Predictor


def test_la_red_tiene_100_capas():
    assert contar_capas(ResNet100()) == 100


@pytest.mark.parametrize("tamano", [32, 64])
def test_forward_produce_logits_por_clase(tamano):
    modelo = ResNet100(num_clases=7).eval()
    with torch.no_grad():
        salida = modelo(torch.randn(2, 3, tamano, tamano))
    assert salida.shape == (2, 7)


def test_predictor_sin_checkpoint_clasifica_bytes(tmp_path):
    predictor = Predictor(tmp_path / "no_existe.pt", dispositivo="cpu")
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), (10, 120, 200)).save(buffer, format="JPEG")

    resultado = predictor.predecir(buffer.getvalue(), top_k=3)

    assert resultado["modelo_entrenado"] is False
    assert len(resultado["top_k"]) == 3
    assert resultado["clase"] == resultado["top_k"][0]["clase"]
    assert 0.0 <= resultado["confianza"] <= 1.0
