from rest_framework import mixins, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from . import servicios
from .models import ImagenProcesada
from .serializers import ImagenProcesadaSerializer, PrediccionLoteSerializer


class ImagenProcesadaViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """Sube una imagen, la procesa con la CNN y guarda el resultado.

    POST   /api/imagenes/       (multipart: imagen, top_k opcional)
    GET    /api/imagenes/
    GET    /api/imagenes/{id}/
    DELETE /api/imagenes/{id}/
    """

    queryset = ImagenProcesada.objects.all()
    serializer_class = ImagenProcesadaSerializer

    def perform_destroy(self, instance):
        instance.imagen.delete(save=False)
        instance.delete()


class PrediccionView(APIView):
    """Clasifica una o varias imágenes sin almacenarlas.

    POST /api/predicciones/  (multipart: imagenes=<archivo> repetido, top_k opcional)
    """

    def post(self, request):
        serializer = PrediccionLoteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        top_k = serializer.validated_data["top_k"]

        resultados = [
            {"nombre": archivo.name, **servicios.clasificar(archivo, top_k)}
            for archivo in serializer.validated_data["imagenes"]
        ]
        return Response({"resultados": resultados}, status=status.HTTP_200_OK)


class ModeloView(APIView):
    """Información de la red neuronal cargada.

    GET /api/modelo/
    """

    def get(self, request):
        return Response(servicios.predictor().informacion())
