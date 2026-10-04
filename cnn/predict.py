"""Clasifica imágenes desde la línea de comandos.

    python -m cnn.predict foto1.jpg foto2.png --top-k 5
"""

import argparse
import json
from pathlib import Path

from .predictor import Predictor


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predicción con la CNN ResNet-100")
    parser.add_argument("imagenes", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    predictor = Predictor(args.checkpoint)
    for ruta in args.imagenes:
        resultado = predictor.predecir(ruta, top_k=args.top_k)
        print(json.dumps({"imagen": str(ruta), **resultado}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
