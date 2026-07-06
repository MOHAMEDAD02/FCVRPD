"""
=============================================================================
CELLULE À INSÉRER DANS LE NOTEBOOK — Section 2.3 : Chargement instances Bouman
=============================================================================
Copiez ce bloc dans une cellule de code juste après la cellule generate_instance().
=============================================================================
"""

# =============================================================================
# 2.3  Chargement d'instances Bouman → VRPInstance avec TFN
# =============================================================================
# Les fichiers Bouman (doublecenter-*.txt) définissent :
#   - la vitesse du camion (v_truck) et du drone (v_drone)
#   - le dépôt et n clients en coordonnées cartésiennes
#   - le type de chaque client : v = camion seul, u = drone-eligible
#
# On convertit en VRPInstance en appliquant exactement les mêmes spreads TFN
# que dans generate_instance() :
#   Camion : TFN(0.85·t, t, 1.30·t)     (variabilité trafic ±15/30 %)
#   Drone  : TFN(0.75·t, t, 1.40·t)     (variabilité vent   ±25/40 %)
#   Autonomie drone : TFN(0.85·A, A, 1.20·A)   avec A = 40 % du rayon max
# =============================================================================

import math, re
from pathlib import Path

BOUMAN_DIR = Path("BOUMAN")          # répertoire contenant les .txt

# ── Spreads TFN (même valeurs que generate_instance) ──────────────────────
_TC_LO, _TC_HI = 0.85, 1.30         # camion
_TD_LO, _TD_HI = 0.75, 1.40         # drone
_AD_LO, _AD_HI = 0.85, 1.20         # autonomie


def parse_bouman_file(path):
    """Retourne (v_truck, v_drone, depot_xy, clients).
    clients = list[(x, y, name, is_drone_eligible)]
    """
    text = re.sub(r"/\*.*?\*/", "", Path(path).read_text(), flags=re.DOTALL)
    it   = iter(text.split())
    v_truck = float(next(it))
    v_drone = float(next(it))
    _n      = int(next(it))          # nombre de clients déclaré dans le fichier
    dx, dy  = float(next(it)), float(next(it))
    next(it)                         # "depot"
    clients = []
    while True:
        try:
            x, y, name = float(next(it)), float(next(it)), next(it)
            clients.append((x, y, name, name.startswith("u")))
        except StopIteration:
            break
    return v_truck, v_drone, (dx, dy), clients


def load_bouman_instance(path, seed=42, Qc=None, Qd=None,
                          demand_min=1, demand_max=5,
                          A_drone_nominal=None,
                          n_trucks=None, n_drones=None):
    """
    Charge un fichier Bouman et retourne un VRPInstance avec TFN.

    Paramètres
    ----------
    path            : chemin du fichier .txt
    seed            : graine pour les demandes aléatoires
    Qc, Qd          : capacités (auto-calibrées si None)
    demand_min/max  : plage de la demande uniforme ∈ [min, max]
    A_drone_nominal : autonomie drone nominale en unités de coordonnées
                      (défaut : 40 % de la distance max de l'instance)
    n_trucks, n_drones : nombre de véhicules (calculés si None)
    """
    rng = np.random.default_rng(seed)
    v_truck, v_drone, depot_xy, clients = parse_bouman_file(path)
    n_clients = len(clients)
    N = n_clients + 1

    # Coordonnées : index 0 = dépôt
    coords = np.zeros((N, 2))
    coords[0] = depot_xy
    drone_eligible = [False]
    for k, (x, y, _name, is_uav) in enumerate(clients, start=1):
        coords[k] = (x, y)
        drone_eligible.append(is_uav)

    # Distances euclidiennes
    def eucl(i, j):
        return math.hypot(coords[i,0]-coords[j,0], coords[i,1]-coords[j,1])
    dist = [[eucl(i,j) for j in range(N)] for i in range(N)]

    # Autonomie drone
    if A_drone_nominal is None:
        max_d = max(dist[i][j] for i in range(N) for j in range(N) if i != j)
        A_drone_nominal = 0.40 * max_d
    A_drone = TFN(
        round(_AD_LO * A_drone_nominal, 6),
        round(         A_drone_nominal, 6),
        round(_AD_HI * A_drone_nominal, 6),
    )

    # Matrices de temps flous
    def truck_tfn(d):
        if d == 0.0: return TFN_ZERO
        t = d / v_truck
        return TFN(round(_TC_LO*t,6), round(t,6), round(_TC_HI*t,6))

    def drone_tfn(d):
        if d == 0.0: return TFN_ZERO
        t = d / v_drone
        return TFN(round(_TD_LO*t,6), round(t,6), round(_TD_HI*t,6))

    time_truck = [[truck_tfn(dist[i][j]) for j in range(N)] for i in range(N)]
    time_drone = [[drone_tfn(dist[i][j]) for j in range(N)] for i in range(N)]

    # Demandes aléatoires
    demands = [0.0] + [float(rng.integers(demand_min, demand_max+1))
                       for _ in range(n_clients)]
    total_q = sum(demands)

    # Capacités auto-calibrées
    if Qc is None:
        avg_route = max(3, n_clients // 3)
        Qc = float(math.ceil(total_q / n_clients * avg_route))
    if Qd is None:
        Qd = float(math.ceil(total_q / n_clients * 3))

    # Flotte
    if n_trucks is None:
        n_trucks = max(2, math.ceil(total_q / Qc))
    if n_drones is None:
        n_drones = max(2, math.ceil(total_q / Qd))

    inst = VRPInstance(
        n_clients=n_clients, n_trucks=n_trucks, n_drones=n_drones,
        Qc=Qc, Qd=Qd, Ad=A_drone,
        demand=demands,
        time_truck=time_truck,
        time_drone=time_drone,
        coords=coords,
    )
    inst.drone_eligible = drone_eligible  # attribut extra : v/u par client
    return inst


# ── Démonstration : instance n=10 (doublecenter-55-n10) ───────────────────
BOUMAN_FILE = BOUMAN_DIR / "doublecenter-55-n10.txt"
INST_BOUMAN = load_bouman_instance(BOUMAN_FILE, seed=42)

print(f"Instance Bouman chargée : {BOUMAN_FILE.name}")
print(f"  n = {INST_BOUMAN.n_clients} clients | "
      f"{INST_BOUMAN.n_trucks} camions (Qc={INST_BOUMAN.Qc:.0f}) | "
      f"{INST_BOUMAN.n_drones} drones (Qd={INST_BOUMAN.Qd:.0f})")
print(f"  Autonomie drone : {INST_BOUMAN.Ad}")
print(f"  Demande totale  : {sum(INST_BOUMAN.demand):.0f}")
n_uav = sum(INST_BOUMAN.drone_eligible[1:])
print(f"  Clients u (drone) : {n_uav} | v (camion seul) : {INST_BOUMAN.n_clients - n_uav}")

# Vérification d'un TFN de temps camion et drone
i, j = 0, 1
print(f"\n  Temps camion 0→1 : {INST_BOUMAN.tC_tfn(i,j)}")
print(f"  Temps drone  0→1 : {INST_BOUMAN.tD_tfn(i,j)}")
