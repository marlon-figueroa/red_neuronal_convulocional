"""Entrena la ResNet-100.

Ejemplos:
    python -m cnn.train --epocas 50
    python -m cnn.train --dataset carpeta --ruta-datos ./mis_imagenes --tamano-imagen 64
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from .config import CHECKPOINT_POR_DEFECTO, DESVIACION_CIFAR10, MEDIA_CIFAR10, TAMANO_IMAGEN
from .data import crear_dataloaders
from .model import ResNet100, contar_capas, contar_parametros
from .utils import seleccionar_dispositivo


def entrenar_epoca(modelo, loader, criterio, optimizador, dispositivo, limite_lotes=None):
    modelo.train()
    perdida_total, aciertos, total = 0.0, 0, 0
    for i, (imagenes, etiquetas) in enumerate(loader):
        if limite_lotes is not None and i >= limite_lotes:
            break
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)

        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * imagenes.size(0)
        aciertos += (salidas.argmax(1) == etiquetas).sum().item()
        total += imagenes.size(0)
    return perdida_total / max(total, 1), aciertos / max(total, 1)


@torch.inference_mode()
def evaluar(modelo, loader, criterio, dispositivo, limite_lotes=None):
    modelo.eval()
    perdida_total, aciertos, total = 0.0, 0, 0
    for i, (imagenes, etiquetas) in enumerate(loader):
        if limite_lotes is not None and i >= limite_lotes:
            break
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        salidas = modelo(imagenes)
        perdida_total += criterio(salidas, etiquetas).item() * imagenes.size(0)
        aciertos += (salidas.argmax(1) == etiquetas).sum().item()
        total += imagenes.size(0)
    return perdida_total / max(total, 1), aciertos / max(total, 1)


def guardar_checkpoint(ruta: Path, modelo, clases, tamano_imagen, ancho_base, precision):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "estado_modelo": modelo.state_dict(),
            "clases": clases,
            "tamano_imagen": tamano_imagen,
            "ancho_base": ancho_base,
            "media": MEDIA_CIFAR10,
            "desviacion": DESVIACION_CIFAR10,
            "precision_validacion": precision,
        },
        ruta,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Entrenamiento de la CNN ResNet-100")
    parser.add_argument("--dataset", choices=["cifar10", "carpeta"], default="cifar10")
    parser.add_argument("--ruta-datos", type=Path, default=None)
    parser.add_argument("--tamano-imagen", type=int, default=TAMANO_IMAGEN)
    parser.add_argument("--ancho-base", type=int, default=16)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--lote", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--trabajadores", type=int, default=2)
    parser.add_argument("--dispositivo", default="auto", help="auto, cpu, cuda o mps")
    parser.add_argument("--salida", type=Path, default=CHECKPOINT_POR_DEFECTO)
    parser.add_argument(
        "--limite-lotes",
        type=int,
        default=None,
        help="Procesa solo N lotes por época (útil para pruebas rápidas)",
    )
    args = parser.parse_args(argv)

    dispositivo = seleccionar_dispositivo(args.dispositivo)
    loader_entrenamiento, loader_validacion, clases = crear_dataloaders(
        args.dataset, args.ruta_datos, args.tamano_imagen, args.lote, args.trabajadores
    )

    modelo = ResNet100(num_clases=len(clases), ancho_base=args.ancho_base).to(dispositivo)
    print(
        f"Modelo ResNet-100 | capas: {contar_capas(modelo)} | "
        f"parámetros: {contar_parametros(modelo):,} | dispositivo: {dispositivo}"
    )

    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.SGD(
        modelo.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=args.epocas)

    mejor_precision = -1.0
    for epoca in range(1, args.epocas + 1):
        inicio = time.time()
        p_ent, a_ent = entrenar_epoca(
            modelo, loader_entrenamiento, criterio, optimizador, dispositivo, args.limite_lotes
        )
        p_val, a_val = evaluar(modelo, loader_validacion, criterio, dispositivo, args.limite_lotes)
        planificador.step()

        print(
            f"Época {epoca:03d}/{args.epocas} | "
            f"pérdida ent {p_ent:.4f} prec ent {a_ent:.2%} | "
            f"pérdida val {p_val:.4f} prec val {a_val:.2%} | "
            f"{time.time() - inicio:.1f}s"
        )

        if a_val > mejor_precision:
            mejor_precision = a_val
            guardar_checkpoint(
                args.salida, modelo, clases, args.tamano_imagen, args.ancho_base, a_val
            )
            print(f"  -> checkpoint guardado en {args.salida}")

    print(f"Mejor precisión de validación: {mejor_precision:.2%}")


if __name__ == "__main__":
    main()
