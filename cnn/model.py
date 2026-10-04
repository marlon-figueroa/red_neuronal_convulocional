"""Red neuronal convolucional residual de 100 capas (ResNet-100).

Conteo de capas con pesos (Conv2d + Linear):
    1 convolución inicial
  + 49 bloques residuales x 2 convoluciones = 98
  + 1 capa totalmente conectada
  = 100 capas

Los atajos (shortcuts) usan la "opción A" del paper de ResNet: submuestreo y
relleno de canales con ceros, sin parámetros, por lo que no suman capas.
"""

import torch
import torch.nn.functional as F
from torch import nn

BLOQUES_POR_ETAPA = (12, 12, 13, 12)
CAPAS_ESPERADAS = 100


class AtajoSinParametros(nn.Module):
    def __init__(self, canales_entrada: int, canales_salida: int, paso: int):
        super().__init__()
        self.paso = paso
        self.relleno = canales_salida - canales_entrada

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.paso > 1:
            x = x[:, :, :: self.paso, :: self.paso]
        if self.relleno > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.relleno))
        return x


class BloqueResidual(nn.Module):
    def __init__(self, canales_entrada: int, canales_salida: int, paso: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(canales_entrada, canales_salida, 3, paso, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(canales_salida)
        self.conv2 = nn.Conv2d(canales_salida, canales_salida, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(canales_salida)

        if paso != 1 or canales_entrada != canales_salida:
            self.atajo = AtajoSinParametros(canales_entrada, canales_salida, paso)
        else:
            self.atajo = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        salida = F.relu(self.bn1(self.conv1(x)))
        salida = self.bn2(self.conv2(salida))
        return F.relu(salida + self.atajo(x))


class ResNet100(nn.Module):
    def __init__(self, num_clases: int = 10, canales_entrada: int = 3, ancho_base: int = 16):
        super().__init__()
        self.conv_inicial = nn.Conv2d(canales_entrada, ancho_base, 3, 1, 1, bias=False)
        self.bn_inicial = nn.BatchNorm2d(ancho_base)

        etapas = []
        canales = ancho_base
        for indice, num_bloques in enumerate(BLOQUES_POR_ETAPA):
            canales_salida = ancho_base * (2**indice)
            paso = 1 if indice == 0 else 2
            bloques = []
            for i in range(num_bloques):
                bloques.append(BloqueResidual(canales, canales_salida, paso if i == 0 else 1))
                canales = canales_salida
            etapas.append(nn.Sequential(*bloques))
        self.etapas = nn.Sequential(*etapas)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.clasificador = nn.Linear(canales, num_clases)

        self._inicializar_pesos()

    def _inicializar_pesos(self) -> None:
        for modulo in self.modules():
            if isinstance(modulo, nn.Conv2d):
                nn.init.kaiming_normal_(modulo.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(modulo, nn.BatchNorm2d):
                nn.init.ones_(modulo.weight)
                nn.init.zeros_(modulo.bias)
            elif isinstance(modulo, nn.Linear):
                nn.init.normal_(modulo.weight, 0, 0.01)
                nn.init.zeros_(modulo.bias)
        # La última BN de cada bloque en cero hace que cada bloque empiece como
        # identidad, lo que estabiliza el entrenamiento de redes tan profundas.
        for modulo in self.modules():
            if isinstance(modulo, BloqueResidual):
                nn.init.zeros_(modulo.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_inicial(self.conv_inicial(x)))
        x = self.etapas(x)
        x = torch.flatten(self.pool(x), 1)
        return self.clasificador(x)


def contar_capas(modelo: nn.Module) -> int:
    return sum(isinstance(m, (nn.Conv2d, nn.Linear)) for m in modelo.modules())


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters())
