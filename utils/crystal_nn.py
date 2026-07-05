"""
crystal_nn.py
=============

Simulador de colocación/unión de prismas 3D que deriva, a partir de la
geometría resultante, una arquitectura de red neuronal (número de capas,
neuronas por capa, capa de entrada y capa de salida).

Flujo:

    Geometría (prismas) -> Cristal 3D -> Algoritmo geométrico
        -> Capas ordenadas -> CrystalArchitecture -> (opcional) modelo PyTorch

Idea central
------------
- Cada prisma es un prisma regular de N lados (por defecto hexagonal),
  definido por su centro (x, y, z), radio, altura, rotación y número de lados.
- Dos prismas están "unidos" si comparten al menos una arista (2+ vértices
  coincidentes dentro de una tolerancia), lo cual se detecta puramente con
  geometría (sin librerías especiales, solo numpy).
- Los "niveles" (capas de la red) se determinan por la coordenada Z de cada
  prisma (se puede cambiar el criterio de agrupación).
- El número de neuronas de cada capa = número de vértices únicos (no
  duplicados) que aparecen en ese nivel.
- El "orden de propagación" dentro de cada nivel se calcula en sentido
  horario alrededor del centroide del clúster (expansión radial).
- Se puede añadir un prisma nuevo en cualquier posición del grupo y
  recalcular toda la arquitectura desde cero (no hay estado cacheado).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Prisma individual
# ---------------------------------------------------------------------------

@dataclass
class Prism:
    id: int
    center: np.ndarray          # [x, y, z]
    radius: float = 1.0         # radio de la base (circunferencia circunscrita)
    height: float = 1.0         # altura del prisma
    rotation: float = 0.0       # rotación en radianes alrededor del eje Z
    sides: int = 6              # nº de lados de la base (6 = hexagonal)

    def _ring_vertices(self, z: float) -> np.ndarray:
        angles = self.rotation + 2 * np.pi * np.arange(self.sides) / self.sides
        xs = self.center[0] + self.radius * np.cos(angles)
        ys = self.center[1] + self.radius * np.sin(angles)
        zs = np.full(self.sides, z)
        return np.stack([xs, ys, zs], axis=1)

    def top_vertices(self) -> np.ndarray:
        return self._ring_vertices(self.center[2] + self.height / 2)

    def bottom_vertices(self) -> np.ndarray:
        return self._ring_vertices(self.center[2] - self.height / 2)

    def all_vertices(self) -> np.ndarray:
        return np.vstack([self.top_vertices(), self.bottom_vertices()])


# ---------------------------------------------------------------------------
# 2. Arquitectura derivada (independiente de cualquier framework)
# ---------------------------------------------------------------------------

@dataclass
class CrystalArchitecture:
    layer_keys: List[float]                 # clave de nivel (p.ej. Z redondeada)
    neurons: List[int]                      # nº de neuronas por capa
    layer_prism_ids: List[List[int]]        # qué prismas componen cada capa
    coordinates: Dict[int, List[float]]     # centro de cada prisma
    adjacency: Dict[int, List[int]]         # grafo de uniones entre prismas
    propagation_order: List[int]            # orden de recorrido (horario, radial)

    # -- utilidades de lectura -------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.neurons)

    @property
    def input_size(self) -> int:
        return self.neurons[0]

    @property
    def output_size(self) -> int:
        return self.neurons[-1]

    @property
    def hidden_sizes(self) -> List[int]:
        return self.neurons[1:-1]

    def summary(self) -> str:
        lines = ["Arquitectura derivada del cristal", "=" * 34]
        lines.append(f"Total de capas: {self.n_layers}")
        lines.append(f"Capa de entrada : {self.input_size} neuronas "
                      f"(nivel z={self.layer_keys[0]}, prismas={self.layer_prism_ids[0]})")
        for i, n in enumerate(self.hidden_sizes, start=1):
            z = self.layer_keys[i]
            pids = self.layer_prism_ids[i]
            lines.append(f"Capa oculta {i:<2}: {n} neuronas (nivel z={z}, prismas={pids})")
        lines.append(f"Capa de salida  : {self.output_size} neuronas "
                      f"(nivel z={self.layer_keys[-1]}, prismas={self.layer_prism_ids[-1]})")
        lines.append(f"Orden de propagación (ids de prisma): {self.propagation_order}")
        return "\n".join(lines)

    # -- compiladores a frameworks --------------------------------------------
    def to_pytorch(self, activation: str = "relu"):
        """Compila la arquitectura a un nn.Sequential de PyTorch."""
        import torch.nn as nn

        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
        }
        act_cls = act_map.get(activation, nn.ReLU)

        modules = []
        for i in range(len(self.neurons) - 1):
            modules.append(nn.Linear(self.neurons[i], self.neurons[i + 1]))
            if i < len(self.neurons) - 2:   # sin activación tras la última capa
                modules.append(act_cls())
        return nn.Sequential(*modules)

    def to_layer_spec(self) -> List[Tuple[int, int]]:
        """Devuelve pares (in_features, out_features) por capa, agnóstico de framework."""
        return [(self.neurons[i], self.neurons[i + 1]) for i in range(len(self.neurons) - 1)]


# ---------------------------------------------------------------------------
# 3. Clúster de prismas (el "cristal")
# ---------------------------------------------------------------------------

class PrismCluster:
    """
    Contenedor de prismas que permite:
      - añadir prismas en cualquier posición,
      - calcular la adyacencia (uniones) entre ellos,
      - calcular los niveles/capas,
      - calcular el orden de propagación,
      - construir (y recalcular) la CrystalArchitecture resultante.
    """

    def __init__(self, vertex_round_decimals: int = 4, z_round_decimals: int = 3):
        self.prisms: Dict[int, Prism] = {}
        self._next_id = 0
        self.vertex_round_decimals = vertex_round_decimals
        self.z_round_decimals = z_round_decimals

    # -- construcción -----------------------------------------------------
    def add_prism(
        self,
        center,
        radius: float = 1.0,
        height: float = 1.0,
        rotation: float = 0.0,
        sides: int = 6,
        warn_if_disconnected: bool = True,
    ) -> int:
        """Añade un prisma en cualquier parte del grupo. Devuelve su id."""
        pid = self._next_id
        prism = Prism(pid, np.array(center, dtype=float), radius, height, rotation, sides)
        self.prisms[pid] = prism
        self._next_id += 1

        if warn_if_disconnected and len(self.prisms) > 1:
            adjacency = self.compute_adjacency()
            if not adjacency.get(pid):
                print(f"[aviso] El prisma {pid} en {tuple(center)} no comparte arista "
                      f"con ningún otro prisma del grupo (queda aislado).")
        return pid

    def remove_prism(self, pid: int) -> None:
        self.prisms.pop(pid, None)

    # -- geometría ----------------------------------------------------------
    def _vertex_key(self, v: np.ndarray) -> Tuple[float, float, float]:
        return tuple(np.round(v, self.vertex_round_decimals))

    def compute_adjacency(self) -> Dict[int, List[int]]:
        """
        Dos prismas son vecinos ("unidos") si comparten al menos 2 vértices
        (es decir, una arista), dentro de la tolerancia de redondeo.
        """
        vertex_map: Dict[Tuple, List[int]] = {}
        for pid, prism in self.prisms.items():
            for v in prism.all_vertices():
                key = self._vertex_key(v)
                vertex_map.setdefault(key, []).append(pid)

        shared_count: Dict[Tuple[int, int], int] = {}
        for pids in vertex_map.values():
            unique_pids = sorted(set(pids))
            for a, b in itertools.combinations(unique_pids, 2):
                shared_count[(a, b)] = shared_count.get((a, b), 0) + 1

        adjacency: Dict[int, set] = {pid: set() for pid in self.prisms}
        for (a, b), count in shared_count.items():
            if count >= 2:
                adjacency[a].add(b)
                adjacency[b].add(a)

        return {pid: sorted(neigh) for pid, neigh in adjacency.items()}

    def compute_levels(self) -> Dict[float, List[int]]:
        """Agrupa prismas por nivel (Z redondeada). Devuelve niveles ordenados asc."""
        levels: Dict[float, List[int]] = {}
        for pid, prism in self.prisms.items():
            z_key = round(float(prism.center[2]), self.z_round_decimals)
            levels.setdefault(z_key, []).append(pid)
        return dict(sorted(levels.items()))

    def propagation_order(self) -> List[int]:
        """
        Orden de recorrido: por nivel (Z ascendente) y, dentro de cada nivel,
        en sentido horario alrededor del centroide del clúster (expansión radial).
        """
        if not self.prisms:
            return []
        levels = self.compute_levels()
        centroid_xy = np.mean([p.center[:2] for p in self.prisms.values()], axis=0)

        order: List[int] = []
        for _, pids in levels.items():
            def angle_key(pid: int) -> float:
                dx, dy = self.prisms[pid].center[:2] - centroid_xy
                angle = np.arctan2(dy, dx)
                return -angle  # negativo => sentido horario

            order.extend(sorted(pids, key=angle_key))
        return order

    # -- arquitectura ---------------------------------------------------------
    def build_architecture(self) -> CrystalArchitecture:
        """Recalcula toda la arquitectura desde cero a partir del estado actual."""
        if not self.prisms:
            raise ValueError("El clúster no tiene prismas todavía.")

        levels = self.compute_levels()
        adjacency = self.compute_adjacency()
        prop_order = self.propagation_order()

        layer_keys, neurons, layer_prism_ids = [], [], []
        for z, pids in levels.items():
            vertex_keys = set()
            for pid in pids:
                for v in self.prisms[pid].all_vertices():
                    vertex_keys.add(self._vertex_key(v))
            layer_keys.append(z)
            neurons.append(len(vertex_keys))
            layer_prism_ids.append(sorted(pids))

        coordinates = {pid: prism.center.tolist() for pid, prism in self.prisms.items()}

        return CrystalArchitecture(
            layer_keys=layer_keys,
            neurons=neurons,
            layer_prism_ids=layer_prism_ids,
            coordinates=coordinates,
            adjacency=adjacency,
            propagation_order=prop_order,
        )


# ---------------------------------------------------------------------------
# 4. Demo / ejemplo de uso
# ---------------------------------------------------------------------------

def hex_neighbor(center: Tuple[float, float, float], angle_deg: float, radius: float = 1.0):
    """
    Devuelve el centro de un hexágono regular (circunradio `radius`) que
    comparte una arista con `center`, en la dirección `angle_deg`.
    Para hexágonos regulares, la distancia entre centros vecinos que
    comparten arista es radius * sqrt(3), y las direcciones válidas (con
    rotación=0) son 30, 90, 150, 210, 270 y 330 grados.
    """
    d = radius * np.sqrt(3)
    a = np.radians(angle_deg)
    return (center[0] + d * np.cos(a), center[1] + d * np.sin(a), center[2])


if __name__ == "__main__":
    cluster = PrismCluster()
    R = 1.0

    # Nivel 0 (entrada): 3 prismas hexagonales mutuamente unidos por arista, en z=0
    A = (0.0, 0.0, 0.0)
    B = hex_neighbor(A, 30, radius=R)   # comparte arista con A
    C = hex_neighbor(A, 90, radius=R)   # comparte arista con A y con B (60° entre ambos)
    cluster.add_prism(center=A, radius=R, height=1.0, sides=6)
    cluster.add_prism(center=B, radius=R, height=1.0, sides=6)
    cluster.add_prism(center=C, radius=R, height=1.0, sides=6)

    # Nivel 1: 2 prismas apilados directamente sobre B y C (comparten cara top/bottom -> z=1)
    cluster.add_prism(center=(B[0], B[1], 1.0), radius=R, height=1.0, sides=6)
    cluster.add_prism(center=(C[0], C[1], 1.0), radius=R, height=1.0, sides=6)

    # Nivel 2 (salida): 1 prisma apilado sobre el anterior (z=2)
    cluster.add_prism(center=(B[0], B[1], 2.0), radius=R, height=1.0, sides=6)

    arch = cluster.build_architecture()
    print(arch.summary())
    print()
    print("Especificación de capas (in, out):", arch.to_layer_spec())

    # --- Ahora se añade un prisma nuevo en cualquier parte del grupo ---
    # Ejemplo: un hexágono que comparte arista con el prisma B del nivel 1 (z=1)
    print("\n--- Añadiendo un nuevo prisma en el nivel 1 ---\n")
    D = hex_neighbor((B[0], B[1], 1.0), 330, radius=R)
    cluster.add_prism(center=D, radius=R, height=1.0, sides=6)

    arch2 = cluster.build_architecture()
    print(arch2.summary())

    # --- Compilación a PyTorch (si está disponible) ---
    try:
        model = arch2.to_pytorch()
        print("\nModelo PyTorch generado:")
        print(model)
    except ImportError:
        print("\n(PyTorch no está instalado en este entorno; se omite la compilación a nn.Sequential)")
