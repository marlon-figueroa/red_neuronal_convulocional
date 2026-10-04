# Red neuronal convolucional de 100 capas + API REST

Proyecto en Python con:

- **`cnn/`**: una red neuronal convolucional residual de **100 capas** (ResNet-100) en PyTorch, con scripts de entrenamiento y predicción.
- **`api/`**: una API REST con **Django REST Framework** que recibe imágenes y las procesa con la red.
- **`cliente/`**: un cliente de línea de comandos (y clase reutilizable `ClienteCNN`) que envía imágenes a la API.

## Arquitectura de la red

| Componente | Capas con pesos |
|---|---|
| Convolución inicial 3x3 | 1 |
| 49 bloques residuales (2 convoluciones 3x3 cada uno), en 4 etapas de 12, 12, 13 y 12 bloques | 98 |
| Capa totalmente conectada (clasificador) | 1 |
| **Total** | **100** |

Cada convolución va seguida de BatchNorm y ReLU. Las conexiones residuales permiten entrenar una red tan profunda; los atajos entre etapas no tienen parámetros (submuestreo + relleno de canales con ceros), por lo que el conteo es exactamente 100 capas `Conv2d` + `Linear`. Usa `AdaptiveAvgPool2d`, así que acepta imágenes de cualquier tamaño.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entrenamiento

Con CIFAR-10 (se descarga automáticamente en `data/`):

```bash
python -m cnn.train --epocas 50
```

Con tus propias imágenes, organizadas en formato ImageFolder:

```
mis_imagenes/
  train/<clase>/*.jpg
  val/<clase>/*.jpg
```

```bash
python -m cnn.train --dataset carpeta --ruta-datos mis_imagenes --tamano-imagen 64 --epocas 30
```

El mejor modelo se guarda en `checkpoints/resnet100.pt` (cámbialo con `--salida`). Detecta automáticamente CUDA, MPS (Apple Silicon) o CPU. Para una prueba rápida usa `--limite-lotes 5`.

Predicción local sin API:

```bash
python -m cnn.predict foto.jpg --top-k 5
```

## API REST (Django REST Framework)

```bash
cd api
python manage.py migrate
python manage.py runserver
```

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/modelo/` | Información de la red (capas, parámetros, clases, dispositivo) |
| `POST` | `/api/imagenes/` | Sube una imagen (`imagen`, `top_k` opcional), la procesa y guarda el resultado |
| `GET` | `/api/imagenes/` | Lista paginada de imágenes procesadas |
| `GET` | `/api/imagenes/{id}/` | Detalle de una imagen procesada |
| `DELETE` | `/api/imagenes/{id}/` | Elimina la imagen y su resultado |
| `POST` | `/api/predicciones/` | Clasifica una o varias imágenes (`imagenes` repetido, máx. 32) sin guardarlas |

Ejemplo con `curl`:

```bash
curl -F "imagen=@foto.jpg" -F "top_k=5" http://127.0.0.1:8000/api/imagenes/
```

Respuesta:

```json
{
  "id": 1,
  "imagen": "http://127.0.0.1:8000/media/imagenes/2026/10/03/foto.jpg",
  "clase": "gato",
  "confianza": 0.91,
  "resultados": [{"clase": "gato", "probabilidad": 0.91}, ...],
  "modelo_entrenado": true,
  "creado": "2026-10-03T23:06:58-06:00"
}
```

Si no existe un checkpoint entrenado, la API funciona con pesos aleatorios y responde `"modelo_entrenado": false` (las predicciones no tienen significado hasta entrenar).

Variables de entorno:

| Variable | Por defecto | Uso |
|---|---|---|
| `CNN_CHECKPOINT` | `checkpoints/resnet100.pt` | Ruta del modelo entrenado |
| `CNN_TAMANO_MAXIMO_IMAGEN_MB` | `10` | Tamaño máximo por imagen |
| `DJANGO_SECRET_KEY` | clave de desarrollo | Cámbiala en producción |
| `DJANGO_DEBUG` | `1` | `0` en producción |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Hosts permitidos |

## Cliente

```bash
python cliente/cliente.py info
python cliente/cliente.py procesar foto.jpg --top-k 5
python cliente/cliente.py predecir foto1.jpg foto2.png
python cliente/cliente.py listar
python cliente/cliente.py obtener 1
python cliente/cliente.py eliminar 1
```

Usa `--url` o la variable `CNN_API_URL` para apuntar a otro servidor. Desde código:

```python
from cliente.cliente import ClienteCNN

cliente = ClienteCNN("http://127.0.0.1:8000/api")
print(cliente.procesar("foto.jpg", top_k=3))
```

## Pruebas

```bash
pytest                                   # red neuronal
cd api && python manage.py test          # API
```
