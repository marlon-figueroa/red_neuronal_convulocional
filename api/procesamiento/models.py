from django.db import models


class ImagenProcesada(models.Model):
    imagen = models.ImageField(upload_to="imagenes/%Y/%m/%d/")
    clase = models.CharField(max_length=100)
    confianza = models.FloatField()
    resultados = models.JSONField(default=list, help_text="Top-k de clases con su probabilidad")
    modelo_entrenado = models.BooleanField(default=False)
    creado = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-creado"]
        verbose_name = "imagen procesada"
        verbose_name_plural = "imágenes procesadas"

    def __str__(self) -> str:
        return f"{self.imagen.name} -> {self.clase} ({self.confianza:.2%})"
