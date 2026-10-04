"""Cliente de línea de comandos para la API REST de la CNN.

Ejemplos:
    python cliente/cliente.py info
    python cliente/cliente.py procesar foto.jpg --top-k 5
    python cliente/cliente.py predecir foto1.jpg foto2.png
    python cliente/cliente.py listar
    python cliente/cliente.py obtener 3
    python cliente/cliente.py eliminar 3
"""

import argparse
import json
import mimetypes
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import requests

URL_POR_DEFECTO = os.environ.get("CNN_API_URL", "http://127.0.0.1:8000/api")


class ClienteCNN:
    def __init__(self, url_base: str = URL_POR_DEFECTO, timeout: float = 60):
        self.url_base = url_base.rstrip("/")
        self.timeout = timeout
        self.sesion = requests.Session()

    def _url(self, ruta: str) -> str:
        return f"{self.url_base}/{ruta.lstrip('/')}"

    @staticmethod
    def _respuesta(respuesta: requests.Response):
        if not respuesta.ok:
            try:
                detalle = respuesta.json()
            except ValueError:
                detalle = respuesta.text
            raise RuntimeError(f"Error HTTP {respuesta.status_code}: {detalle}")
        return respuesta.json() if respuesta.content else None

    @staticmethod
    def _archivo(ruta: Path, abiertos: ExitStack):
        tipo = mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
        return ruta.name, abiertos.enter_context(open(ruta, "rb")), tipo

    def informacion_modelo(self) -> dict:
        return self._respuesta(self.sesion.get(self._url("modelo/"), timeout=self.timeout))

    def procesar(self, ruta: str | Path, top_k: int = 3) -> dict:
        """Envía una imagen para procesarla y guardarla en el servidor."""
        with ExitStack() as abiertos:
            archivos = {"imagen": self._archivo(Path(ruta), abiertos)}
            respuesta = self.sesion.post(
                self._url("imagenes/"), files=archivos, data={"top_k": top_k}, timeout=self.timeout
            )
        return self._respuesta(respuesta)

    def predecir(self, rutas: list[str | Path], top_k: int = 3) -> dict:
        """Envía una o varias imágenes para clasificarlas sin guardarlas."""
        with ExitStack() as abiertos:
            archivos = [("imagenes", self._archivo(Path(r), abiertos)) for r in rutas]
            respuesta = self.sesion.post(
                self._url("predicciones/"),
                files=archivos,
                data={"top_k": top_k},
                timeout=self.timeout,
            )
        return self._respuesta(respuesta)

    def listar(self, pagina: int = 1) -> dict:
        return self._respuesta(
            self.sesion.get(self._url("imagenes/"), params={"page": pagina}, timeout=self.timeout)
        )

    def obtener(self, identificador: int) -> dict:
        return self._respuesta(
            self.sesion.get(self._url(f"imagenes/{identificador}/"), timeout=self.timeout)
        )

    def eliminar(self, identificador: int) -> None:
        self._respuesta(
            self.sesion.delete(self._url(f"imagenes/{identificador}/"), timeout=self.timeout)
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cliente de la API de la CNN ResNet-100")
    parser.add_argument("--url", default=URL_POR_DEFECTO, help="URL base de la API")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("info", help="Información del modelo")

    p_procesar = sub.add_parser("procesar", help="Procesa y guarda una imagen")
    p_procesar.add_argument("imagen", type=Path)
    p_procesar.add_argument("--top-k", type=int, default=3)

    p_predecir = sub.add_parser("predecir", help="Clasifica imágenes sin guardarlas")
    p_predecir.add_argument("imagenes", nargs="+", type=Path)
    p_predecir.add_argument("--top-k", type=int, default=3)

    p_listar = sub.add_parser("listar", help="Lista imágenes procesadas")
    p_listar.add_argument("--pagina", type=int, default=1)

    p_obtener = sub.add_parser("obtener", help="Detalle de una imagen procesada")
    p_obtener.add_argument("id", type=int)

    p_eliminar = sub.add_parser("eliminar", help="Elimina una imagen procesada")
    p_eliminar.add_argument("id", type=int)

    args = parser.parse_args(argv)
    cliente = ClienteCNN(args.url)

    try:
        if args.comando == "info":
            resultado = cliente.informacion_modelo()
        elif args.comando == "procesar":
            resultado = cliente.procesar(args.imagen, args.top_k)
        elif args.comando == "predecir":
            resultado = cliente.predecir(args.imagenes, args.top_k)
        elif args.comando == "listar":
            resultado = cliente.listar(args.pagina)
        elif args.comando == "obtener":
            resultado = cliente.obtener(args.id)
        else:
            cliente.eliminar(args.id)
            resultado = {"eliminado": args.id}
    except (RuntimeError, requests.RequestException, OSError) as error:
        print(error, file=sys.stderr)
        return 1

    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
