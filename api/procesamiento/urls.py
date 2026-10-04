from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import ImagenProcesadaViewSet, ModeloView, PrediccionView

router = DefaultRouter()
router.register("imagenes", ImagenProcesadaViewSet, basename="imagen")

urlpatterns = [
    path("", include(router.urls)),
    path("predicciones/", PrediccionView.as_view(), name="predicciones"),
    path("modelo/", ModeloView.as_view(), name="modelo"),
]
