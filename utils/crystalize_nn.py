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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

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
# 2. (Opcional) Ejemplo de otra figura: Tetraedro
# ---------------------------------------------------------------------------

@dataclass
class TetrahedronCell(GeometricCell):
    id: int
    center: np.ndarray
    scale: float = 1.0

    def all_vertices(self) -> np.ndarray:
        # Tetraedro regular centrado en `center` con arista `scale`
        # Vértices de un tetraedro regular con centro en origen
        s = self.scale / np.sqrt(6)  # para que la arista sea scale
        vertices = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ]) * s
        return vertices + self.center


# ---------------------------------------------------------------------------
# 3. Arquitectura derivada (independiente de cualquier framework)
# ---------------------------------------------------------------------------

@dataclass
class CrystalArchitecture:
    layer_keys: List[float]                 # clave de nivel (p.ej. Z redondeada)
    neurons: List[int]                      # nº de neuronas por capa
    layer_vertices: List[Set[Tuple[float, float, float]]]  # vértices únicos por capa
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
# 4. Clúster de prismas (el "cristal")
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
        self.cells: Dict[int, GeometricCell] = {}
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
        """
        Añade un prisma (o cualquier GeometricCell) al clúster.
        Si el primer argumento es una instancia de GeometricCell, se añade directamente.
        En caso contrario, crea un Prism con los parámetros dados.
        Devuelve el id de la celda añadida.
        """
        if isinstance(center, GeometricCell):
            cell = center
            # Si la celda no tiene id, se le asigna uno
            if cell.id is None:
                cell.id = self._next_id
            pid = cell.id
            self.cells[pid] = cell
            self._next_id = max(self._next_id, pid + 1)
        else:
            pid = self._next_id
            prism = Prism(pid, np.array(center, dtype=float), radius, height, rotation, sides)
            self.cells[pid] = prism
            self._next_id += 1

        if warn_if_disconnected and len(self.cells) > 1:
            adjacency = self.compute_adjacency()
            if not adjacency.get(pid):
                print(f"[aviso] La celda {pid} en {tuple(center)} no comparte arista "
                      f"con ninguna otra celda del grupo (queda aislada).")
        return pid

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
        vertex_map: Dict[Tuple, List[int]] = {}
        for pid, cell in self.cells.items():
            for v in cell.all_vertices():
                key = self._vertex_key(v)
                vertex_map.setdefault(key, []).append(pid)

        shared_count: Dict[Tuple[int, int], int] = {}
        for pids in vertex_map.values():
            unique_pids = sorted(set(pids))
            for a, b in itertools.combinations(unique_pids, 2):
                shared_count[(a, b)] = shared_count.get((a, b), 0) + 1

        adjacency: Dict[int, set] = {pid: set() for pid in self.cells}
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
            vertex_set = set()
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
        except ImportError:
            raise ImportError(
                "CadQuery no está instalado. Por favor instálelo con: pip install cadquery"
            )

        result = cq.Workplane("XY")
        for cell in self.cells.values():
            if isinstance(cell, Prism):
                # Crear polígono base (en el plano XY, centro en origen)
                angles = cell.rotation + 2 * np.pi * np.arange(cell.sides) / cell.sides
                pts = [(cell.radius * np.cos(a), cell.radius * np.sin(a)) for a in angles]
                wp = cq.Workplane("XY").polygon(pts).extrude(cell.height)
                # Trasladar para que el centro del prisma coincida con cell.center
                # (la base inferior está en cell.center[2] - height/2)
                wp = wp.translate(
                    (cell.center[0], cell.center[1], cell.center[2] - cell.height / 2)
                )
                result = result.union(wp)
            elif isinstance(cell, TetrahedronCell):
                # Obtener vértices (4 puntos) y caras triangulares
                vertices = cell.all_vertices()  # (4,3)
                # Orden de caras para un tetraedro: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
                faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
                try:
                    solid = cq.Solid.makePolyhedron(vertices.tolist(), faces)
                    result = result.union(solid)
                except Exception as e:
                    warnings.warn(f"Error al construir tetraedro {cell.id}: {e}. Se omite.")
            else:
                warnings.warn(
                    f"Celda tipo {type(cell).__name__} (id {cell.id}) no soportada para exportación STEP."
                )

        # Exportar al archivo
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

    def __init__(self, cluster: PrismCluster, architecture: CrystalArchitecture,
                 model: Optional['torch.nn.Module'] = None):
        """
        Args:
            cluster: el clúster geométrico actual.
            architecture: la arquitectura derivada del clúster.
            model: modelo PyTorch (opcional). Si no se da, se construye con
                   `architecture.to_pytorch()`.
        """
        self.cluster = cluster
        self.architecture = architecture
        if model is None:
            model = architecture.to_pytorch()
        self.model = model
        self._optimizer = None  # se puede asignar después con attach_optimizer

    def attach_optimizer(self, optimizer: 'torch.optim.Optimizer'):
        """Asocia un optimizador al modelo actual."""
        self._optimizer = optimizer

    def grow(self, new_cell_or_params, **kwargs) -> Tuple['torch.nn.Module', CrystalArchitecture]:
        """
        Añade una nueva celda al clúster y actualiza el modelo preservando la función.
        Devuelve (nuevo_modelo, nueva_arquitectura).
        """
        # 1. Añadir la celda al clúster
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
            self._update_optimizer()

        return new_model, new_arch

    def _compute_diff(self, old_arch: CrystalArchitecture,
                      new_arch: CrystalArchitecture) -> Dict:
        """
        Compara dos arquitecturas y devuelve un diccionario con las diferencias.
        """
        diff = {
            'new_layers': [],       # índices donde se insertan nuevas capas
            'expanded_layers': [],  # (índice, nuevas_neuronas, índices_antiguos_a_duplicar)
            'unchanged': []
        }

        # Comparar capas por clave Z (asumimos que las claves están ordenadas)
        old_keys = old_arch.layer_keys
        new_keys = new_arch.layer_keys

        # Si hay nuevas claves (capas insertadas)
        # (asumimos que solo se añaden, no se eliminan)
        old_set = set(old_keys)
        new_set = set(new_keys)
        added_keys = sorted(new_set - old_set)
        # Ubicar en qué posiciones aparecen
        for key in added_keys:
            idx = new_keys.index(key)
            diff['new_layers'].append(idx)

        # Para capas existentes, comparar conjuntos de vértices
        old_vertices = old_arch.layer_vertices
        new_vertices = new_arch.layer_vertices
        # Mapeo de clave a índice en la nueva
        key_to_idx_new = {k: i for i, k in enumerate(new_keys)}
        for i, (old_key, old_vert_set) in enumerate(zip(old_keys, old_vertices)):
            if old_key not in key_to_idx_new:
                # Esta capa desapareció (no debería ocurrir)
                warnings.warn(f"La capa con clave {old_key} ha desaparecido.")
                continue
            j = key_to_idx_new[old_key]
            new_vert_set = new_vertices[j]
            if new_vert_set == old_vert_set:
                diff['unchanged'].append(j)
            elif old_vert_set.issubset(new_vert_set):
                # Ensanchamiento
                added_vertices = new_vert_set - old_vert_set
                n_new = len(added_vertices)
                # Elegir neuronas antiguas a duplicar (aleatoriamente, o la última)
                # Para simplicidad, duplicamos las últimas `n_new` neuronas de la capa
                old_neuron_indices = list(range(len(old_vert_set)))
                # Ordenamos los vértices para tener un orden determinista
                old_vert_list = sorted(old_vert_set)
                new_vert_list = sorted(new_vert_set)
                # Índices de neuronas nuevas en la nueva capa (posiciones finales)
                # (asumimos que las nuevas se añaden al final)
                # Para saber qué índices corresponden a las nuevas, comparar las listas
                # (asumimos que el orden es consistente)
                # En una implementación real, necesitaríamos un mapeo estable.
                # Aquí, simplemente duplicamos las últimas n_new neuronas.
                n_old = len(old_vert_list)
                n_new_total = len(new_vert_list)
                # Los índices de las nuevas neuronas son n_old .. n_new_total-1
                new_indices = list(range(n_old, n_new_total))
                # Decidir qué neuronas antiguas duplicar (por ejemplo, las últimas n_new)
                old_indices_to_duplicate = list(range(n_old - n_new, n_old))
                diff['expanded_layers'].append((j, n_new, old_indices_to_duplicate))
            else:
                # Caso complejo: ni subconjunto ni superconjunto (no debería ocurrir)
                warnings.warn(f"Cambio no trivial en capa {old_key}. No se maneja.")
                diff['unchanged'].append(j)

        # Ordenar expanded_layers por índice para aplicar de derecha a izquierda
        diff['expanded_layers'].sort(key=lambda x: x[0], reverse=True)
        diff['new_layers'].sort(reverse=True)

        return diff

    def _apply_growth(self, model: 'torch.nn.Module',
                      diff: Dict) -> 'torch.nn.Module':
        """
        Aplica las diferencias al modelo secuencial.
        """
        import torch.nn as nn
        if not isinstance(model, nn.Sequential):
            raise TypeError("Solo se soportan modelos nn.Sequential para crecimiento.")

        modules = list(model.children())

        # 1. Insertar nuevas capas (si las hay)
        for idx in diff['new_layers']:
            # No implementado: lanzar error
            raise NotImplementedError("Inserción de capas no implementada aún.")

        # 2. Ensanchar capas existentes (aplicar de atrás hacia adelante)
        for layer_idx, n_new, old_indices in diff['expanded_layers']:
            # La capa layer_idx es un Linear (y la siguiente también puede ser Linear)
            # Modificar la capa layer_idx (out_features aumenta)
            # y la capa layer_idx+1 (in_features aumenta, si existe)
            if layer_idx >= len(modules):
                continue
            layer = modules[layer_idx]
            if not isinstance(layer, nn.Linear):
                warnings.warn(f"La capa {layer_idx} no es Linear, no se puede ensanchar.")
                continue

            # Obtener dimensiones
            in_features = layer.in_features
            out_features = layer.out_features
            new_out = out_features + n_new

            # Duplicar pesos y bias de las neuronas seleccionadas
            weight = layer.weight.data  # (out, in)
            bias = layer.bias.data if layer.bias is not None else None

            # Elegir filas a duplicar (las que corresponden a old_indices)
            # Duplicamos esas filas y las añadimos al final
            dup_rows = weight[old_indices]  # (n_new, in)
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

            # Reemplazar la capa en la lista
            modules[layer_idx] = new_layer

            # Ahora actualizar la siguiente capa (si existe) para que tenga in_features = new_out
            if layer_idx + 1 < len(modules):
                next_layer = modules[layer_idx + 1]
                if isinstance(next_layer, nn.Linear):
                    # Aumentar sus in_features
                    old_in = next_layer.in_features
                    old_out = next_layer.out_features
                    # Duplicar columnas correspondientes a las neuronas duplicadas
                    weight_next = next_layer.weight.data  # (out, old_in)
                    # Las columnas a duplicar son las mismas old_indices
                    # (porque corresponden a las neuronas de salida de la capa anterior)
                    dup_cols = weight_next[:, old_indices]  # (out, n_new)
                    new_weight_next = torch.cat([weight_next, dup_cols], dim=1)
                    # El bias no cambia
                    new_next = nn.Linear(new_out, old_out, bias=(next_layer.bias is not None))
                    new_next.weight.data = new_weight_next
                    if next_layer.bias is not None:
                        new_next.bias.data = next_layer.bias.data
                    modules[layer_idx + 1] = new_next

        # Reconstruir el Sequential
        new_model = nn.Sequential(*modules)
        return new_model

    def _update_optimizer(self):
        """
        Actualiza el optimizador asociado para incluir los nuevos parámetros,
        preservando el estado de los parámetros antiguos.
        """
        if self._optimizer is None:
            return

        import torch
        old_optimizer = self._optimizer
        old_state = old_optimizer.state_dict()

        # Crear un nuevo optimizador con los mismos hiperparámetros
        new_optimizer = type(old_optimizer)(
            self.model.parameters(),
            **{k: v for k, v in old_optimizer.defaults.items() if k != 'params'}
        )

        # Intentar cargar el estado antiguo para los parámetros que coincidan
        # Nota: esto funciona si los parámetros tienen los mismos nombres
        # o si están en el mismo orden. Como hemos modificado el modelo,
        # los parámetros nuevos aparecerán al final.
        # Tomamos el estado del nuevo optimizador (vacío) y lo fusionamos.
        new_state = new_optimizer.state_dict()
        # Las claves de 'state' son números de índice que corresponden a los
        # parámetros en el orden actual. Para mapear, usamos los id de los tensores.
        old_params = list(old_optimizer.param_groups[0]['params'])
        new_params = list(new_optimizer.param_groups[0]['params'])

        # Construir un mapeo de id de tensor antiguo a su estado
        old_state_dict = old_state['state']
        id_to_state = {}
        for idx, p in enumerate(old_params):
            if idx in old_state_dict:
                id_to_state[id(p)] = old_state_dict[idx]

        # Ahora, para cada nuevo parámetro, si su id está en id_to_state, copiamos
        # el estado; si no, lo inicializamos a ceros (o a un estado por defecto)
        new_state_dict = {}
        for idx, p in enumerate(new_params):
            if id(p) in id_to_state:
                new_state_dict[idx] = id_to_state[id(p)]
            else:
                # Inicializar estado (exp, exp_avg, etc.) a cero
                # Depende del optimizador; para Adam, por ejemplo:
                if isinstance(old_optimizer, torch.optim.Adam):
                    new_state_dict[idx] = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(p),
                        'exp_avg_sq': torch.zeros_like(p),
                    }
                else:
                    # Fallback: estado vacío
                    new_state_dict[idx] = {}

        new_state['state'] = new_state_dict
        new_optimizer.load_state_dict(new_state)
        self._optimizer = new_optimizer


# ---------------------------------------------------------------------------
# 6. Política de crecimiento (esqueleto)
# ---------------------------------------------------------------------------

class GrowthPolicy:
    """Decide cuándo y cómo crecer basado en métricas de entrenamiento."""
    def __init__(self, growable: GrowableModel, patience: int = 10,
                 min_delta: float = 1e-4):
        self.growable = growable
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, loss: float) -> bool:
        """Retorna True si se debe crecer."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                return True
            return False


# ---------------------------------------------------------------------------
# 7. Demo / ejemplo de uso
# ---------------------------------------------------------------------------

def hex_neighbor(center: Tuple[float, float, float], angle_deg: float, radius: float = 1.0):
    """
    Devuelve el centro de un hexágono regular (circunradio `radius`) que
    comparte una arista con `center`, en la dirección `angle_deg`.
    """
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

    # Nivel 0 (entrada): 3 prismas hexagonales mutuamente unidos por arista, en z=0
    A = (0.0, 0.0, 0.0)
    B = hex_neighbor(A, 30, radius=R)
    C = hex_neighbor(A, 90, radius=R)
    cluster.add_prism(center=A, radius=R, height=1.0, sides=6)
    cluster.add_prism(center=B, radius=R, height=1.0, sides=6)
    cluster.add_prism(center=C, radius=R, height=1.0, sides=6)

    # Nivel 1: 2 prismas apilados directamente sobre B y C (z=1)
    cluster.add_prism(center=(B[0], B[1], 1.0), radius=R, height=1.0, sides=6)
    cluster.add_prism(center=(C[0], C[1], 1.0), radius=R, height=1.0, sides=6)

    # Nivel 2 (salida): 1 prisma apilado sobre el anterior (z=2)
    cluster.add_prism(center=(B[0], B[1], 2.0), radius=R, height=1.0, sides=6)

    arch = cluster.build_architecture()
    print(arch.summary())
    print()
    print("Especificación de capas (in, out):", arch.to_layer_spec())

    # --- Exportar a STEP (si CadQuery está instalado) ---
    try:
        cluster.export_step("crystal_initial.step")
    except ImportError as e:
        print(f"Exportación STEP omitida: {e}")

    # --- Crear modelo PyTorch ---
    model = arch.to_pytorch()
    print("\nModelo inicial:")
    print(model)

    # --- Simular entrenamiento y crecimiento ---
    # Creamos un GrowableModel
    growable = GrowableModel(cluster, arch, model)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    growable.attach_optimizer(optimizer)

    # Datos ficticios (entrada = vector de tamaño input_size, salida = vector de tamaño output_size)
    input_size = arch.input_size
    output_size = arch.output_size
    x = torch.randn(10, input_size)
    y = torch.randn(10, output_size)
    criterion = nn.MSELoss()

    # Entrenamiento de ejemplo
    for epoch in range(20):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:2d}, loss = {loss.item():.6f}")

        # Cada 5 épocas, añadimos un prisma en el nivel 1 (ensanchamiento)
        if (epoch + 1) % 5 == 0:
            print(f"\n--- Crecimiento en época {epoch+1} ---")
            # Añadir un hexágono en z=1 que comparte arista con el prisma B de ese nivel
            D = hex_neighbor((B[0], B[1], 1.0), 330, radius=R)
            # Antes de crecer, guardamos el estado del optimizador (opcional)
            # Realizar crecimiento
            new_model, new_arch = growable.grow(center=D, radius=R, height=1.0, sides=6)
            model = new_model  # actualizar referencia
            print("Nueva arquitectura:")
            print(new_arch.summary())
            # El optimizador ya fue actualizado internamente
            print("Optimizador actualizado.\n")

    print("\nModelo final:")
    print(model)
    print("\nArquitectura final:")
    print(growable.architecture.summary())

    # --- Exportar STEP final ---
    try:
        cluster.export_step("crystal_final.step")
    except ImportError as e:
        print(f"Exportación STEP omitida: {e}")