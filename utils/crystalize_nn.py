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

Nuevas características (adaptación)
-----------------------------------
- Abstracción `GeometricCell` que permite usar cualquier figura (prisma,
  tetraedro, etc.) siempre que implemente `all_vertices()` y `center`.
- `GrowableModel`: envoltorio que permite crecer la red durante el
  entrenamiento preservando la función (Net2Net) para ensanchamiento de capas.
- Soporte para crecimiento manual o automático (esqueleto de `GrowthPolicy`).
- El modelo puede ser secuencial o basado en grafo (se deja preparado).
- Exportación a STEP (CAD) mediante CadQuery (`export_step`).
"""

from __future__ import annotations

import itertools
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# 0. Interfaz de celda geométrica
# ---------------------------------------------------------------------------

class GeometricCell(ABC):
    """Interfaz para cualquier figura que pueda colocarse en el cristal."""

    id: int
    center: np.ndarray

    @abstractmethod
    def all_vertices(self) -> np.ndarray:
        """Devuelve un array (N, 3) con todos los vértices de la figura."""
        pass


# ---------------------------------------------------------------------------
# 1. Prisma individual (hereda de GeometricCell)
# ---------------------------------------------------------------------------

@dataclass
class Prism(GeometricCell):
    id: int
    center: np.ndarray  # [x, y, z]
    radius: float = 1.0  # radio de la base (circunferencia circunscrita)
    height: float = 1.0  # altura del prisma
    rotation: float = 0.0  # rotación en radianes alrededor del eje Z
    sides: int = 6  # nº de lados de la base (6 = hexagonal)

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
# 2. (Opcional) Ejemplo de otra figura: Tetraedro
# ---------------------------------------------------------------------------

@dataclass
class TetrahedronCell(GeometricCell):
    id: int
    center: np.ndarray
    scale: float = 1.0

    def all_vertices(self) -> np.ndarray:
        # Tetraedro regular centrado en `center` con arista `scale`
        s = self.scale / np.sqrt(6)  # para que la arista sea scale
        vertices = np.array(
            [
                [1, 1, 1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
            ]
        ) * s
        return vertices + self.center


# ---------------------------------------------------------------------------
# 3. Arquitectura derivada (independiente de cualquier framework)
# ---------------------------------------------------------------------------

@dataclass
class CrystalArchitecture:
    layer_keys: List[float]  # clave de nivel (p.ej. Z redondeada)
    neurons: List[int]  # nº de neuronas por capa
    layer_vertices: List[Set[Tuple[float, float, float]]]  # vértices únicos por capa
    layer_prism_ids: List[List[int]]  # qué prismas componen cada capa
    coordinates: Dict[int, List[float]]  # centro de cada prisma
    adjacency: Dict[int, List[int]]  # grafo de uniones entre prismas
    propagation_order: List[int]  # orden de recorrido (horario, radial)

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
        lines.append(
            f"Capa de entrada : {self.input_size} neuronas "
            f"(nivel z={self.layer_keys[0]}, prismas={self.layer_prism_ids[0]})"
        )
        for i, n in enumerate(self.hidden_sizes, start=1):
            z = self.layer_keys[i]
            pids = self.layer_prism_ids[i]
            lines.append(
                f"Capa oculta {i:<2}: {n} neuronas (nivel z={z}, prismas={pids})"
            )
        lines.append(
            f"Capa de salida  : {self.output_size} neuronas "
            f"(nivel z={self.layer_keys[-1]}, prismas={self.layer_prism_ids[-1]})"
        )
        lines.append(f"Orden de propagación (ids de prisma): {self.propagation_order}")
        return "\n".join(lines)

    def to_pytorch(self, activation: str = "relu") -> "torch.nn.Sequential":
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
            if i < len(self.neurons) - 2:  # sin activación tras la última capa
                modules.append(act_cls())
        return nn.Sequential(*modules)

    def to_layer_spec(self) -> List[Tuple[int, int]]:
        """Devuelve pares (in_features, out_features) por capa, agnóstico de framework."""
        return [(self.neurons[i], self.neurons[i + 1]) for i in range(len(self.neurons) - 1)]


# ---------------------------------------------------------------------------
# 4. Clúster de prismas (el "cristal")
# ---------------------------------------------------------------------------

class PrismCluster:
    """
    Contenedor de celdas geométricas que permite:
      - añadir prismas o cualquier GeometricCell,
      - calcular la adyacencia (uniones) entre ellos,
      - calcular los niveles/capas,
      - calcular el orden de propagación,
      - construir (y recalcular) la CrystalArchitecture resultante.
    """

    def __init__(self, vertex_round_decimals: int = 4, z_round_decimals: int = 3) -> None:
        self.cells: Dict[int, GeometricCell] = {}
        self._next_id = 0
        self.vertex_round_decimals = vertex_round_decimals
        self.z_round_decimals = z_round_decimals

    # -- construcción -----------------------------------------------------
    def add_prism(
        self,
        center: Union[Tuple[float, float, float], List[float], np.ndarray],
        radius: float = 1.0,
        height: float = 1.0,
        rotation: float = 0.0,
        sides: int = 6,
        warn_if_disconnected: bool = True,
    ) -> int:
        """Añade un prisma al clúster y devuelve su id."""
        pid = self._next_id
        self._next_id += 1
        prism = Prism(pid, np.array(center, dtype=float), radius, height, rotation, sides)
        self.cells[pid] = prism
        self._check_connection(pid, warn_if_disconnected)
        return pid

    def add_cell(self, cell: GeometricCell, warn_if_disconnected: bool = True) -> int:
        """Añade una celda ya creada (debe tener id) al clúster."""
        if cell.id is None:  # type: ignore[attr-defined]
            cell.id = self._next_id
            self._next_id += 1
        pid = cell.id
        self.cells[pid] = cell
        self._next_id = max(self._next_id, pid + 1)
        self._check_connection(pid, warn_if_disconnected)
        return pid

    def _check_connection(self, pid: int, warn: bool) -> None:
        if warn and len(self.cells) > 1:
            adjacency = self.compute_adjacency()
            if not adjacency.get(pid):
                print(
                    f"[aviso] La celda {pid} en {tuple(self.cells[pid].center)} "
                    "no comparte arista con ninguna otra celda del grupo (queda aislada)."
                )

    def remove_cell(self, pid: int) -> None:
        self.cells.pop(pid, None)

    # -- geometría ----------------------------------------------------------
    def _vertex_key(self, v: np.ndarray) -> Tuple[float, float, float]:
        return tuple(np.round(v, self.vertex_round_decimals))

    def compute_adjacency(self) -> Dict[int, List[int]]:
        """
        Dos celdas son vecinas ("unidas") si comparten al menos 2 vértices
        (es decir, una arista), dentro de la tolerancia de redondeo.
        """
        vertex_map: Dict[Tuple[float, float, float], List[int]] = {}
        for pid, cell in self.cells.items():
            for v in cell.all_vertices():
                key = self._vertex_key(v)
                vertex_map.setdefault(key, []).append(pid)

        shared_count: Dict[Tuple[int, int], int] = {}
        for pids in vertex_map.values():
            unique_pids = sorted(set(pids))
            for a, b in itertools.combinations(unique_pids, 2):
                shared_count[(a, b)] = shared_count.get((a, b), 0) + 1

        adjacency: Dict[int, Set[int]] = {pid: set() for pid in self.cells}
        for (a, b), count in shared_count.items():
            if count >= 2:
                adjacency[a].add(b)
                adjacency[b].add(a)

        return {pid: sorted(neigh) for pid, neigh in adjacency.items()}

    def compute_levels(self) -> Dict[float, List[int]]:
        """Agrupa celdas por nivel (Z redondeada). Devuelve niveles ordenados asc."""
        levels: Dict[float, List[int]] = {}
        for pid, cell in self.cells.items():
            z_key = round(float(cell.center[2]), self.z_round_decimals)
            levels.setdefault(z_key, []).append(pid)
        return dict(sorted(levels.items()))

    def propagation_order(self) -> List[int]:
        """
        Orden de recorrido: por nivel (Z ascendente) y, dentro de cada nivel,
        en sentido horario alrededor del centroide del clúster (expansión radial).
        """
        if not self.cells:
            return []
        levels = self.compute_levels()
        centroid_xy = np.mean([cell.center[:2] for cell in self.cells.values()], axis=0)

        order: List[int] = []
        for _, pids in levels.items():
            # Ordenar en sentido horario (ángulo negativo)
            def angle_key(pid: int) -> float:
                dx, dy = self.cells[pid].center[:2] - centroid_xy
                angle = np.arctan2(dy, dx)
                return -angle  # negativo => sentido horario

            order.extend(sorted(pids, key=angle_key))
        return order

    # -- arquitectura ---------------------------------------------------------
    def build_architecture(self) -> CrystalArchitecture:
        """Recalcula toda la arquitectura desde cero a partir del estado actual."""
        if not self.cells:
            raise ValueError("El clúster no tiene celdas todavía.")

        levels = self.compute_levels()
        adjacency = self.compute_adjacency()
        prop_order = self.propagation_order()

        layer_keys, neurons, layer_vertices, layer_prism_ids = [], [], [], []
        for z, pids in levels.items():
            vertex_set: Set[Tuple[float, float, float]] = set()
            for pid in pids:
                for v in self.cells[pid].all_vertices():
                    vertex_set.add(self._vertex_key(v))
            layer_keys.append(z)
            neurons.append(len(vertex_set))
            layer_vertices.append(vertex_set)
            layer_prism_ids.append(sorted(pids))

        coordinates = {pid: cell.center.tolist() for pid, cell in self.cells.items()}

        return CrystalArchitecture(
            layer_keys=layer_keys,
            neurons=neurons,
            layer_vertices=layer_vertices,
            layer_prism_ids=layer_prism_ids,
            coordinates=coordinates,
            adjacency=adjacency,
            propagation_order=prop_order,
        )

    # -- exportación CAD (STEP) -----------------------------------------------
    def export_step(self, filename: str) -> None:
        """
        Exporta la geometría del clúster a un archivo STEP (.step o .stp)
        usando CadQuery. Requiere que CadQuery esté instalado.
        """
        try:
            import cadquery as cq
        except ImportError as e:
            raise ImportError(
                "CadQuery no está instalado. Por favor instálelo con: pip install cadquery"
            ) from e

        result = cq.Workplane("XY")
        for cell in self.cells.values():
            if isinstance(cell, Prism):
                radius = float(cell.radius)
                sides = int(cell.sides)
                height = float(cell.height)
                wp = cq.Workplane("XY").polygon(sides, radius).extrude(height)
                wp = wp.translate(
                    (
                        float(cell.center[0]),
                        float(cell.center[1]),
                        float(cell.center[2]) - height / 2,
                    )
                )
                result = result.union(wp)
            elif isinstance(cell, TetrahedronCell):
                vertices = cell.all_vertices()  # (4,3)
                faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
                try:
                    solid = cq.Solid.makePolyhedron(vertices.tolist(), faces)
                    wp = cq.Workplane().add(solid)
                    result = result.union(wp)
                except Exception as e:
                    warnings.warn(f"Error al construir tetraedro {cell.id}: {e}. Se omite.")
            else:
                warnings.warn(
                    f"Celda tipo {type(cell).__name__} (id {cell.id}) no soportada para exportación STEP."
                )

        cq.exporters.export(result, filename)
        print(f"Archivo STEP exportado a: {filename}")


# ---------------------------------------------------------------------------
# 5. Modelo crecible (GrowableModel)
# ---------------------------------------------------------------------------

class GrowableModel:
    """
    Envoltorio que mantiene un modelo PyTorch y su arquitectura asociada,
    permitiendo crecer la red añadiendo nuevos prismas sin perder lo aprendido.
    """

    def __init__(
        self,
        cluster: PrismCluster,
        architecture: CrystalArchitecture,
        model: Optional["torch.nn.Module"] = None,
    ) -> None:
        self.cluster = cluster
        self.architecture = architecture
        if model is None:
            model = architecture.to_pytorch()
        self.model = model
        self._optimizer = None

    def attach_optimizer(self, optimizer: "torch.optim.Optimizer") -> None:
        """Asocia un optimizador al modelo actual."""
        self._optimizer = optimizer

    def grow(
        self,
        new_cell_or_params: Union[GeometricCell, Tuple[float, float, float], List[float], np.ndarray],
        **kwargs,
    ) -> Tuple["torch.nn.Module", CrystalArchitecture]:
        """
        Añade una nueva celda al clúster y actualiza el modelo preservando la función.
        Devuelve (nuevo_modelo, nueva_arquitectura).

        Ejemplo de uso:
            model, arch = growable.grow((0,0,1), radius=1.0, height=1.0, sides=6)
        """
        # 1. Añadir la celda al clúster
        if isinstance(new_cell_or_params, GeometricCell):
            pid = self.cluster.add_cell(new_cell_or_params, warn_if_disconnected=False)
        else:
            pid = self.cluster.add_prism(new_cell_or_params, **kwargs)

        # 2. Calcular nueva arquitectura
        new_arch = self.cluster.build_architecture()

        # 3. Calcular diff y aplicar cirugía
        diff = self._compute_diff(self.architecture, new_arch)
        new_model = self._apply_growth(self.model, diff)

        # 4. Actualizar estado interno
        self.model = new_model
        self.architecture = new_arch

        # 5. Si hay optimizador, actualizarlo
        if self._optimizer is not None:
            self._update_optimizer(diff)

        return new_model, new_arch

    def _compute_diff(
        self, old_arch: CrystalArchitecture, new_arch: CrystalArchitecture
    ) -> Dict[str, Any]:
        """
        Compara dos arquitecturas y devuelve un diccionario con las diferencias.

        Estructura del diff:
            'new_layers': [(posición, in_features, out_features)]
            'expanded_layers': [(posición, n_new, old_indices, new_indices)]
            'unchanged': [posiciones de capas sin cambios]
        """
        diff: Dict[str, Any] = {
            "new_layers": [],
            "expanded_layers": [],
            "unchanged": [],
        }

        old_keys = old_arch.layer_keys
        new_keys = new_arch.layer_keys

        old_set = set(old_keys)
        new_set = set(new_keys)
        added_keys = sorted(new_set - old_set)

        key_to_idx_new = {k: i for i, k in enumerate(new_keys)}
        key_to_idx_old = {k: i for i, k in enumerate(old_keys)}

        # Capas nuevas
        for key in added_keys:
            pos = new_keys.index(key)
            n_neurons = new_arch.neurons[pos]
            if pos == 0:
                in_feat = new_arch.neurons[0]
            else:
                in_feat = new_arch.neurons[pos - 1]
            if pos == len(new_arch.neurons) - 1:
                out_feat = new_arch.neurons[-1]
            else:
                out_feat = new_arch.neurons[pos + 1]
            diff["new_layers"].append((pos, in_feat, out_feat))

        # Capas existentes
        common_keys = old_set & new_set
        for key in sorted(common_keys):
            old_idx = key_to_idx_old[key]
            new_idx = key_to_idx_new[key]
            old_vert_set = old_arch.layer_vertices[old_idx]
            new_vert_set = new_arch.layer_vertices[new_idx]

            if old_vert_set == new_vert_set:
                diff["unchanged"].append(new_idx)
            elif old_vert_set.issubset(new_vert_set):
                new_vertices = list(new_vert_set - old_vert_set)
                old_vert_list = sorted(old_vert_set)
                new_vert_list = sorted(new_vert_set)

                old_centroids = [np.array(v) for v in old_vert_list]
                old_indices_to_duplicate = []
                new_indices = []
                for nv in new_vertices:
                    nv_arr = np.array(nv)
                    dists = [np.linalg.norm(nv_arr - old_c) for old_c in old_centroids]
                    closest_old_idx = int(np.argmin(dists))
                    old_indices_to_duplicate.append(closest_old_idx)
                    new_idx_in_layer = new_vert_list.index(nv)
                    new_indices.append(new_idx_in_layer)

                diff["expanded_layers"].append(
                    (new_idx, len(new_vertices), old_indices_to_duplicate, new_indices)
                )
            else:
                warnings.warn(f"Cambio no trivial en capa con clave {key}. No se maneja.")
                diff["unchanged"].append(new_idx)

        diff["expanded_layers"].sort(key=lambda x: x[0], reverse=True)
        diff["new_layers"].sort(key=lambda x: x[0], reverse=True)

        return diff

    def _apply_growth(self, model: "torch.nn.Module", diff: Dict[str, Any]) -> "torch.nn.Module":
        """Aplica las diferencias al modelo secuencial."""
        import torch
        import torch.nn as nn

        if not isinstance(model, nn.Sequential):
            raise TypeError("Solo se soportan modelos nn.Sequential para crecimiento.")

        modules = list(model.children())

        # Insertar nuevas capas (de atrás hacia adelante)
        for pos, in_feat, out_feat in diff["new_layers"]:
            new_layer = nn.Linear(in_feat, out_feat)
            if in_feat == out_feat:
                new_layer.weight.data = torch.eye(in_feat)
            else:
                nn.init.xavier_uniform_(new_layer.weight)
            if new_layer.bias is not None:
                nn.init.zeros_(new_layer.bias)
            modules.insert(pos, new_layer)

        # Ensanchar capas existentes (de atrás hacia adelante)
        for layer_idx, n_new, old_indices, _ in diff["expanded_layers"]:
            if layer_idx >= len(modules):
                continue
            layer = modules[layer_idx]
            if not isinstance(layer, nn.Linear):
                warnings.warn(f"La capa {layer_idx} no es Linear, no se puede ensanchar.")
                continue

            in_features = layer.in_features
            out_features = layer.out_features
            new_out = out_features + n_new

            weight = layer.weight.data
            bias = layer.bias.data if layer.bias is not None else None

            dup_rows = weight[old_indices]
            new_weight = torch.cat([weight, dup_rows], dim=0)
            if bias is not None:
                dup_bias = bias[old_indices]
                new_bias = torch.cat([bias, dup_bias])
            else:
                new_bias = None

            new_layer = nn.Linear(in_features, new_out, bias=(bias is not None))
            new_layer.weight.data = new_weight
            if new_bias is not None:
                new_layer.bias.data = new_bias

            modules[layer_idx] = new_layer

            # Actualizar la siguiente capa (si existe)
            if layer_idx + 1 < len(modules):
                next_layer = modules[layer_idx + 1]
                if isinstance(next_layer, nn.Linear):
                    old_in = next_layer.in_features
                    old_out = next_layer.out_features
                    weight_next = next_layer.weight.data
                    dup_cols = weight_next[:, old_indices]
                    new_weight_next = torch.cat([weight_next, dup_cols], dim=1)
                    new_next = nn.Linear(new_out, old_out, bias=(next_layer.bias is not None))
                    new_next.weight.data = new_weight_next
                    if next_layer.bias is not None:
                        new_next.bias.data = next_layer.bias.data
                    modules[layer_idx + 1] = new_next

        return nn.Sequential(*modules)

    def _update_optimizer(self, diff: Dict[str, Any]) -> None:
        """Actualiza el optimizador para incluir nuevos parámetros, preservando el estado."""
        if self._optimizer is None:
            return

        import torch

        old_optimizer = self._optimizer
        old_state = old_optimizer.state_dict()

        new_optimizer = type(old_optimizer)(
            self.model.parameters(),
            **{k: v for k, v in old_optimizer.defaults.items() if k != "params"},
        )

        new_state = new_optimizer.state_dict()
        old_params = list(old_optimizer.param_groups[0]["params"])
        new_params = list(new_optimizer.param_groups[0]["params"])

        old_state_dict = old_state["state"]
        id_to_state = {}
        for idx, p in enumerate(old_params):
            if idx in old_state_dict:
                id_to_state[id(p)] = old_state_dict[idx]

        new_state_dict = {}
        for idx, p in enumerate(new_params):
            if id(p) in id_to_state:
                new_state_dict[idx] = id_to_state[id(p)]
            else:
                if isinstance(old_optimizer, torch.optim.Adam):
                    new_state_dict[idx] = {
                        "step": 0,
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p),
                    }
                else:
                    new_state_dict[idx] = {}

        new_state["state"] = new_state_dict
        new_optimizer.load_state_dict(new_state)
        self._optimizer = new_optimizer


# ---------------------------------------------------------------------------
# 6. Política de crecimiento (esqueleto)
# ---------------------------------------------------------------------------

class GrowthPolicy:
    """Decide cuándo y cómo crecer basado en métricas de entrenamiento."""

    def __init__(self, growable: GrowableModel, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.growable = growable
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss: float) -> bool:
        """Retorna True si se debe crecer."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.counter = 0
            return True
        return False


# ---------------------------------------------------------------------------
# 7. Demo / ejemplo de uso
# ---------------------------------------------------------------------------

def hex_neighbor(center: Tuple[float, float, float], angle_deg: float, radius: float = 1.0) -> Tuple[float, float, float]:
    """Devuelve el centro de un hexágono vecino en la dirección dada."""
    d = radius * np.sqrt(3)
    a = np.radians(angle_deg)
    return (center[0] + d * np.cos(a), center[1] + d * np.sin(a), center[2])


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # --- Crear clúster inicial ---
    cluster = PrismCluster()
    R = 1.0

    A = (0.0, 0.0, 0.0)
    B = hex_neighbor(A, 30, radius=R)
    C = hex_neighbor(A, 90, radius=R)
    cluster.add_prism(center=A, radius=R, height=1.0, sides=6)
    cluster.add_prism(center=B, radius=R, height=1.0, sides=6)
    cluster.add_prism(center=C, radius=R, height=1.0, sides=6)

    cluster.add_prism(center=(B[0], B[1], 1.0), radius=R, height=1.0, sides=6)
    cluster.add_prism(center=(C[0], C[1], 1.0), radius=R, height=1.0, sides=6)

    cluster.add_prism(center=(B[0], B[1], 2.0), radius=R, height=1.0, sides=6)

    arch = cluster.build_architecture()
    print(arch.summary())
    print()
    print("Especificación de capas (in, out):", arch.to_layer_spec())

    # --- Exportar a STEP (opcional) ---
    try:
        cluster.export_step("crystal_initial.step")
    except ImportError as e:
        print(f"Exportación STEP omitida: {e}")

    # --- Crear modelo PyTorch ---
    model = arch.to_pytorch()
    print("\nModelo inicial:")
    print(model)

    # --- Simular entrenamiento y crecimiento ---
    growable = GrowableModel(cluster, arch, model)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    growable.attach_optimizer(optimizer)

    input_size = arch.input_size
    output_size = arch.output_size
    x = torch.randn(10, input_size)
    y = torch.randn(10, output_size)
    criterion = nn.MSELoss()

    for epoch in range(20):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:2d}, loss = {loss.item():.6f}")

        if (epoch + 1) % 5 == 0:
            print(f"\n--- Crecimiento en época {epoch+1} ---")
            D = hex_neighbor((B[0], B[1], 1.0), 330, radius=R)
            # CORRECCIÓN: pasar D como primer argumento posicional
            new_model, new_arch = growable.grow(D, radius=R, height=1.0, sides=6)
            model = new_model
            print("Nueva arquitectura:")
            print(new_arch.summary())
            print("Optimizador actualizado.\n")

    print("\nModelo final:")
    print(model)
    print("\nArquitectura final:")
    print(growable.architecture.summary())

    try:
        cluster.export_step("crystal_final.step")
    except ImportError as e:
        print(f"Exportación STEP omitida: {e}")