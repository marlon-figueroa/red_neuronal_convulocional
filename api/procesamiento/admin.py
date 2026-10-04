from django.contrib import admin

from .models import ImagenProcesada


@admin.register(ImagenProcesada)
class ImagenProcesadaAdmin(admin.ModelAdmin):
    list_display = ("id", "imagen", "clase", "confianza", "modelo_entrenado", "creado")
    list_filter = ("clase", "modelo_entrenado")
    readonly_fields = ("clase", "confianza", "resultados", "modelo_entrenado", "creado")
