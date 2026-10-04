from django.conf import settings
from rest_framework import serializers

from . import servicios
from .models import ImagenProcesada

MAXIMO_LOTE = 32


def validar_tamano(imagen):
    limite_mb = settings.CNN_TAMANO_MAXIMO_IMAGEN_MB
    if imagen.size > limite_mb * 1024 * 1024:
        raise serializers.ValidationError(f"La imagen supera el límite de {limite_mb} MB.")
    return imagen


class ImagenProcesadaSerializer(serializers.ModelSerializer):
    top_k = serializers.IntegerField(
        write_only=True, required=False, default=3, min_value=1, max_value=20
    )

    class Meta:
        model = ImagenProcesada
        fields = [
            "id",
            "imagen",
            "top_k",
            "clase",
            "confianza",
            "resultados",
            "modelo_entrenado",
            "creado",
        ]
        read_only_fields = ["clase", "confianza", "resultados", "modelo_entrenado", "creado"]

    def validate_imagen(self, imagen):
        return validar_tamano(imagen)

    def create(self, validated_data):
        top_k = validated_data.pop("top_k")
        resultado = servicios.clasificar(validated_data["imagen"], top_k)
        return ImagenProcesada.objects.create(
            **validated_data,
            clase=resultado["clase"],
            confianza=resultado["confianza"],
            resultados=resultado["top_k"],
            modelo_entrenado=resultado["modelo_entrenado"],
        )


class PrediccionLoteSerializer(serializers.Serializer):
    imagenes = serializers.ListField(
        child=serializers.ImageField(validators=[validar_tamano]),
        allow_empty=False,
        max_length=MAXIMO_LOTE,
    )
    top_k = serializers.IntegerField(required=False, default=3, min_value=1, max_value=20)
