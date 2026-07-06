"""
Chargement des instances du benchmark Bouman (FSTSP/doublecenter) pour le CVRPD
avec temps de trajet et autonomie flous (TFN).

Format des fichiers Bouman :
  /*The speed of the Truck*/
  <v_truck>
  /*The speed of the Drone*/
  <v_drone>
  /*Number of Nodes*/
  <n>           ← nombre de clients (hors dépôt)
  /*The Depot*/
  <x> <y> depot
  /*The Locations (x_coor y_coor name)*/
  <x> <y> v<k>  ← client camion seulement  (vehicle-only)
  <x> <y> u<k>  ← client accessible drone  (UAV-eligible)

Les coordonnées sont cartésiennes (unités arbitraires).
Les distances sont euclidiennes ; les spreads TFN reprennent exactement la
méthode du notebook B&B (même facteurs ±).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# TFN — import conditionnel (disponible quand le notebook est chargé)
# ---------------------------------------------------------------------------
try:
    from __main__ import TFN, VRPInstance          # notebook already defined them
except ImportError:
    # stand-alone usage : définitions minimales
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class TFN:                                      # noqa: F811
        a1: float; a2: float; a3: float

        def __post_init__(self):
            if not (self.a1 <= self.a2 + 1e-9 and self.a2 <= self.a3 + 1e-9):
                raise ValueError(f"TFN invalide : ({self.a1},{self.a2},{self.a3})")

        def f1(self): return (self.a1 + 2*self.a2 + self.a3) / 4.0
        def f2(self): return self.a2
        def f3(self): return self.a3 - self.a1
        def lex_key(self): return (self.f1(), self.f2(), self.f3())
        def __add__(self, o):
            return TFN(self.a1+o.a1, self.a2+o.a2, self.a3+o.a3) if isinstance(o,TFN) \
                   else TFN(self.a1+o, self.a2+o, self.a3+o)
        def __mul__(self, s):
            return TFN(self.a1*s, self.a2*s, self.a3*s) if s>=0 \
                   else TFN(self.a3*s, self.a2*s, self.a1*s)
        __rmul__ = __mul__
        def __lt__(self, o): return self.lex_key() < o.lex_key()
        def __le__(self, o): return self.lex_key() <= o.lex_key()
        def __repr__(self): return f"({self.a1:.2f},{self.a2:.2f},{self.a3:.2f})"

    @_dc
    class VRPInstance:                              # noqa: F811
        n_clients: int; n_trucks: int; n_drones: int
        Qc: float; Qd: float; Ad: TFN
        demand: list; time_truck: list; time_drone: list
        coords: np.ndarray = field(default_factory=lambda: np.array([]))
        drone_eligible: list = field(default_factory=list)

        @property
        def n(self): return self.n_clients
        @property
        def mc(self): return self.n_trucks
        @property
        def md(self): return self.n_drones
        @property
        def A_drone(self): return self.Ad
        @property
        def q(self): return np.array(self.demand)
        @property
        def n_nodes(self): return self.n_clients + 1
        def V(self): return list(range(self.n_clients + 1))
        def C(self): return list(range(1, self.n_clients + 1))
        def tC_tfn(self, i, j): t=self.time_truck[i][j]; return TFN(t.a1,t.a2,t.a3)
        def tD_tfn(self, i, j): t=self.time_drone[i][j]; return TFN(t.a1,t.a2,t.a3)


# ---------------------------------------------------------------------------
# Spreads TFN (identiques au notebook B&B)
# ---------------------------------------------------------------------------
# Camion : TFN(0.85·t, t, 1.30·t)  — variabilité trafic ±15/30 %
_TC_LO, _TC_HI = 0.85, 1.30
# Drone  : TFN(0.75·t, t, 1.40·t)  — variabilité vent   ±25/40 %
_TD_LO, _TD_HI = 0.75, 1.40

# Autonomie drone par défaut (unités cohérentes avec les coordonnées)
# Calibrée sur la distance max observée dans les petites instances Bouman.
# Peut être surchargée via le paramètre A_drone_nominal de load_bouman_instance().
_A_DRONE_SPREAD = (0.85, 1.0, 1.20)    # ±15/20 % autour du nominal


# ---------------------------------------------------------------------------
# Parseur
# ---------------------------------------------------------------------------

def _parse_bouman_file(path: str | Path):
    """Retourne (v_truck, v_drone, depot_xy, clients) où
    clients = list of (x, y, name, is_drone_eligible:bool).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # Supprime les commentaires /* … */
    clean = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    tokens = clean.split()

    it = iter(tokens)

    v_truck = float(next(it))
    v_drone = float(next(it))
    n_nodes = int(next(it))          # nombre de clients (hors dépôt dans la plupart des fichiers)

    # Dépôt
    dx, dy = float(next(it)), float(next(it))
    _label = next(it)                # "depot"

    clients = []
    while True:
        try:
            x = float(next(it))
            y = float(next(it))
            name = next(it)
            is_uav = name.startswith("u")
            clients.append((x, y, name, is_uav))
        except StopIteration:
            break

    return v_truck, v_drone, (dx, dy), clients


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def load_bouman_instance(
    path: str | Path,
    seed: int = 42,
    Qc: float = None,
    Qd: float = None,
    demand_min: int = 1,
    demand_max: int = 5,
    A_drone_nominal: float = None,
    n_trucks: int = None,
    n_drones: int = None,
) -> "VRPInstance":
    """
    Charge un fichier Bouman et le convertit en VRPInstance avec TFN.

    Parameters
    ----------
    path : chemin vers le fichier .txt Bouman
    seed : graine pour les demandes aléatoires
    Qc   : capacité camion (défaut : calibrée automatiquement)
    Qd   : capacité drone  (défaut : calibrée automatiquement)
    demand_min/max : plage de la demande uniformément tirée ∈ [min, max]
    A_drone_nominal : autonomie nominale drone en unités de coordonnées
                      (défaut : 40 % de la distance max observée)
    n_trucks/n_drones : fixe le nombre de véhicules (défaut : calculé)
    """
    rng = np.random.default_rng(seed)

    v_truck, v_drone, depot_xy, clients = _parse_bouman_file(path)
    n_clients = len(clients)

    # ── Coordonnées (index 0 = dépôt) ────────────────────────────────────
    coords = np.zeros((n_clients + 1, 2))
    coords[0] = depot_xy
    drone_eligible = [False]           # dépôt n'est pas un client
    for k, (x, y, _name, is_uav) in enumerate(clients, start=1):
        coords[k] = (x, y)
        drone_eligible.append(is_uav)

    N = n_clients + 1

    # ── Distances euclidiennes ────────────────────────────────────────────
    def eucl(i, j):
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        return math.hypot(dx, dy)

    dist = [[eucl(i, j) for j in range(N)] for i in range(N)]

    # ── Autonomie drone par défaut ────────────────────────────────────────
    if A_drone_nominal is None:
        # 40 % de la distance max dans l'instance → couvre la majorité des arcs
        max_d = max(dist[i][j] for i in range(N) for j in range(N) if i != j)
        A_drone_nominal = 0.40 * max_d

    A_drone = TFN(
        round(_A_DRONE_SPREAD[0] * A_drone_nominal, 4),
        round(_A_DRONE_SPREAD[1] * A_drone_nominal, 4),
        round(_A_DRONE_SPREAD[2] * A_drone_nominal, 4),
    )

    # ── Temps flous ───────────────────────────────────────────────────────
    def truck_tfn(d):
        if d == 0.0:
            return TFN(0.0, 0.0, 0.0)
        t = d / v_truck
        return TFN(round(_TC_LO * t, 6), round(t, 6), round(_TC_HI * t, 6))

    def drone_tfn(d):
        if d == 0.0:
            return TFN(0.0, 0.0, 0.0)
        t = d / v_drone
        return TFN(round(_TD_LO * t, 6), round(t, 6), round(_TD_HI * t, 6))

    time_truck = [[truck_tfn(dist[i][j]) for j in range(N)] for i in range(N)]
    time_drone = [[drone_tfn(dist[i][j]) for j in range(N)] for i in range(N)]

    # ── Demandes aléatoires ───────────────────────────────────────────────
    demands = [0.0] + [
        float(rng.integers(demand_min, demand_max + 1)) for _ in range(n_clients)
    ]
    total_demand = sum(demands)

    # ── Capacités auto-calibrées ──────────────────────────────────────────
    if Qc is None:
        # Chaque camion couvre en moyenne ~n/3 clients
        avg_route = max(3, n_clients // 3)
        Qc = float(math.ceil(total_demand / n_clients * avg_route))
    if Qd is None:
        # Drones couvrent ~3 clients chacun (missions courtes)
        Qd = float(math.ceil(total_demand / n_clients * 3))

    # ── Flotte ────────────────────────────────────────────────────────────
    if n_trucks is None:
        n_trucks = max(2, math.ceil(total_demand / Qc))
    if n_drones is None:
        n_drones = max(2, math.ceil(total_demand / Qd))

    inst = VRPInstance(
        n_clients=n_clients,
        n_trucks=n_trucks,
        n_drones=n_drones,
        Qc=Qc,
        Qd=Qd,
        Ad=A_drone,
        demand=demands,
        time_truck=time_truck,
        time_drone=time_drone,
        coords=coords,
    )
    # Attribut supplémentaire pour la distinction v/u (non requis par VRPInstance)
    object.__setattr__(inst, "drone_eligible", drone_eligible) if hasattr(inst, "__dataclass_fields__") \
        else setattr(inst, "drone_eligible", drone_eligible)

    return inst


# ---------------------------------------------------------------------------
# Utilitaire : liste des instances disponibles
# ---------------------------------------------------------------------------

def list_bouman_instances(folder: str | Path = "BOUMAN") -> list[Path]:
    """Retourne les fichiers .txt triés par nombre de clients."""
    folder = Path(folder)
    files = sorted(folder.glob("*.txt"), key=lambda p: (
        int(re.search(r"n(\d+)", p.name).group(1)),
        p.name,
    ))
    return files


# ---------------------------------------------------------------------------
# Affichage rapide
# ---------------------------------------------------------------------------

def display_bouman_instance(inst, title: str = "Instance Bouman"):
    """Affichage textuel + carte (réutilise le style du notebook)."""
    import matplotlib.pyplot as plt

    print("=" * 72)
    print(f"  {title}  —  n = {inst.n_clients} clients")
    print("=" * 72)
    print(f"  Capacité camion Qc = {inst.Qc}")
    print(f"  Capacité drone  Qd = {inst.Qd}")
    print(f"  Flotte           : {inst.n_trucks} camions, {inst.n_drones} drones")
    print(f"  Autonomie drone  : {inst.Ad}")
    print(f"  Demande totale   : {sum(inst.demand):.0f}")

    eligible = getattr(inst, "drone_eligible", [False]*inst.n_nodes)
    n_uav = sum(eligible[1:])
    n_trk = inst.n_clients - n_uav
    print(f"  Clients camion seul (v) : {n_trk}")
    print(f"  Clients drone-eligible (u) : {n_uav}")
    print()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(*inst.coords[0], s=300, marker="s", c="black",
               label="Dépôt", zorder=5)
    ax.annotate("0", inst.coords[0], color="white",
                ha="center", va="center", fontweight="bold", zorder=6)

    colors = ["#F44336" if eligible[i] else "#2196F3"
              for i in inst.C()]
    ax.scatter(inst.coords[1:, 0], inst.coords[1:, 1],
               s=120, c=colors, edgecolor="black", linewidth=1.2,
               zorder=4)
    for i in inst.C():
        ax.annotate(str(i), inst.coords[i], ha="center", va="center",
                    fontsize=7, fontweight="bold", zorder=5)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        plt.scatter([], [], s=80, c="#F44336", edgecolors="black", label="drone-eligible (u)"),
        plt.scatter([], [], s=80, c="#2196F3", edgecolors="black", label="camion seul (v)"),
        plt.scatter([], [], s=150, marker="s", c="black", label="Dépôt"),
    ], loc="upper left", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
