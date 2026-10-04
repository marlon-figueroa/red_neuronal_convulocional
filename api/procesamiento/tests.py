import io
import shutil
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from PIL import Image
from rest_framework import status
from rest_framework.test import APITestCase

from .models import ImagenProcesada

MEDIA_TEMPORAL = tempfile.mkdtemp()


def imagen_prueba(nombre="prueba.png", color=(200, 30, 30), tamano=(64, 48)):
    buffer = io.BytesIO()
    Image.new("RGB", tamano, color).save(buffer, format="PNG")
    return SimpleUploadedFile(nombre, buffer.getvalue(), content_type="image/png")


@override_settings(MEDIA_ROOT=MEDIA_TEMPORAL)
class ApiProcesamientoTests(APITestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(MEDIA_TEMPORAL, ignore_errors=True)
        super().tearDownClass()

    def test_info_modelo_reporta_100_capas(self):
        respuesta = self.client.get(reverse("modelo"))
        self.assertEqual(respuesta.status_code, status.HTTP_200_OK)
        self.assertEqual(respuesta.data["capas"], 100)

    def test_subir_imagen_la_procesa_y_guarda(self):
        respuesta = self.client.post(
            reverse("imagen-list"), {"imagen": imagen_prueba(), "top_k": 5}, format="multipart"
        )
        self.assertEqual(respuesta.status_code, status.HTTP_201_CREATED, respuesta.data)
        self.assertEqual(len(respuesta.data["resultados"]), 5)
        self.assertIn(respuesta.data["clase"], [r["clase"] for r in respuesta.data["resultados"]])
        self.assertEqual(ImagenProcesada.objects.count(), 1)

        detalle = self.client.get(reverse("imagen-detail", args=[respuesta.data["id"]]))
        self.assertEqual(detalle.status_code, status.HTTP_200_OK)

    def test_rechaza_archivo_que_no_es_imagen(self):
        archivo = SimpleUploadedFile("texto.txt", b"no soy una imagen", content_type="text/plain")
        respuesta = self.client.post(reverse("imagen-list"), {"imagen": archivo}, format="multipart")
        self.assertEqual(respuesta.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(ImagenProcesada.objects.count(), 0)

    def test_prediccion_por_lote_sin_guardar(self):
        respuesta = self.client.post(
            reverse("predicciones"),
            {"imagenes": [imagen_prueba("a.png"), imagen_prueba("b.png", (0, 0, 255))]},
            format="multipart",
        )
        self.assertEqual(respuesta.status_code, status.HTTP_200_OK, respuesta.data)
        self.assertEqual([r["nombre"] for r in respuesta.data["resultados"]], ["a.png", "b.png"])
        self.assertEqual(ImagenProcesada.objects.count(), 0)

    def test_eliminar_imagen(self):
        creada = self.client.post(
            reverse("imagen-list"), {"imagen": imagen_prueba()}, format="multipart"
        )
        respuesta = self.client.delete(reverse("imagen-detail", args=[creada.data["id"]]))
        self.assertEqual(respuesta.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(ImagenProcesada.objects.count(), 0)
