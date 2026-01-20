import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# --- Chargement des données (inchangé) ---
data_files = ['../data/midgame/probcut_mid.txt', '../data/midgame/probcut_mid_ext.txt']
data_files_end = ['../data/endgame/probcut_end.txt', '../data/endgame/probcut_end_ext.txt', '../data/endgame/probcut_end_ext2.txt']

data = [[[[] for _ in range(61)] for _ in range(61)] for _ in range(65)]

for data_file in data_files:
    try:
        with open(data_file, 'r') as f:
            raw_data = f.read().splitlines()
        for datum in raw_data:
            n_discs, depth1, depth2, error = [int(elem) for elem in datum.split()]
            if n_discs > 5 and depth2 > 1:
                data[n_discs][depth1][depth2].append(error)
    except:
        print('cannot open', data_file)

for data_file in data_files_end:
    try:
        with open(data_file, 'r') as f:
            raw_data = f.read().splitlines()
        for datum in raw_data:
            n_discs, depth1, error = [int(elem) for elem in datum.split()]
            depth2 = 64 - n_discs
            data[n_discs][depth1][depth2].append(error)
    except:
        print('cannot open', data_file)


# --- Préparation des tableaux ---
w_n_discs_sd = []
x_depth1_sd = []
y_depth2_sd = []
z_sd = []
weight_sd = []

for n_discs in range(6, len(data)):
    for depth1 in range(len(data[n_discs])):
        for depth2 in range(len(data[n_discs][depth1])):
            if len(data[n_discs][depth1][depth2]) >= 10:
                sd = statistics.stdev(data[n_discs][depth1][depth2])
                w_n_discs_sd.append(n_discs)
                x_depth1_sd.append(depth1)
                y_depth2_sd.append(depth2)
                z_sd.append(sd)
                if n_discs == 64-depth2:
                    weight_sd.append(0.05)
                else:
                    weight_sd.append(1)

# --- Ajout des points de guidage ---
for n_discs in range(6, 34):
    depth2 = 30 + (n_discs & 1)
    depth1 = 0 + (n_discs & 1)
    w_n_discs_sd.append(n_discs)
    x_depth1_sd.append(depth1)
    y_depth2_sd.append(depth2)
    z_sd.append(12.0 - 0.143 * ((n_discs+(n_discs & 1))-6))
    weight_sd.append(0.25)

for n_discs in range(6, 25):
    depth2 = 40 + (n_discs & 1)
    depth1 = 0 + (n_discs & 1)
    w_n_discs_sd.append(n_discs)
    x_depth1_sd.append(depth1)
    y_depth2_sd.append(depth2)
    z_sd.append(18.0 - 0.111 * ((n_discs+(n_discs & 1))-6))
    weight_sd.append(0.05)

for n_discs in range(6, 63):
    for depth2 in range(2, 64-n_discs+1):
        depth1 = depth2-2
        w_n_discs_sd.append(n_discs)
        x_depth1_sd.append(depth1)
        y_depth2_sd.append(depth2)
        z_sd.append(2.5)
        weight_sd.append(0.5)

for n_discs in range(6, 65):
    for depth in (0, 64-n_discs+1):
        w_n_discs_sd.append(n_discs)
        x_depth1_sd.append(depth)
        y_depth2_sd.append(depth)
        z_sd.append(0)
        weight_sd.append(2)


# --- Nouvelle fonction : zones internes ---
def sigmoid(x, x0, k=0.5):
    """Fonction sigmoïde utilisée pour adoucir les transitions."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def f_zoned(wxy,
            a1,b1,c1,d1,e1,f1,g1,    # mid
            a2,b2,c2,d2,e2,f2,g2     # end
    ):
    w, x, y = wxy
    w = 64 - np.asarray(w, dtype=float)

    # --- Poids doux selon w ---
    # s1 augmente vers 1 quand w dépasse 30 (transition mid/end)
    s1 = sigmoid(w, 30, k=0.3)

    w_mid = s1                          # poids mid
    w_end = 1 - s1                      # poids end

    # --- Calcul des 2 sous-modèles ---
    def poly(a,b,c,d,e,f,g):
        r = a*w + b*x + c*y
        return d*r**3 + e*r**2 + f*r + g

    f1 = poly(a1,b1,c1,d1,e1,f1,g1)
    f2 = poly(a2,b2,c2,d2,e2,f2,g2)

    # --- Combinaison douce ---
    res = w_mid*f1 + w_end*f2
    return res
    
# --- bornage de la fonction ---
def f_max(wxy, *params):
    return np.minimum(40.0, np.maximum(0.0, f_zoned(wxy, *params)))


# --- Ajustement global avec zones internes ---
p0 = np.ones(14)  # 2 zones × 7 paramètres
popt_sd, pcov_sd = curve_fit(
    f_zoned,
    (w_n_discs_sd, x_depth1_sd, y_depth2_sd),
    z_sd,
    p0,
    sigma=weight_sd,
    absolute_sigma=True,
    maxfev=10000
)

# --- Sortie format C++ : probcut_a[] = {a1, a2, a3}; etc. ---

param_names = ["a", "b", "c", "d", "e", "f", "g"]

# Il y a 3 zones, donc 3 jeux de 7 paramètres
for j, name in enumerate(param_names):
    vals = [popt_sd[j + 7*i] for i in range(2)]  # i = 0→early, 1→mid, 2→end
    vals_str = ", ".join(f"{v:.8f}" for v in vals)
    print(f"constexpr double probcut_{name}[] = {{{vals_str}}};")

# --- Affichage copiable dans un tableur : une valeur par ligne, séparation par zone ---
nb_params = 7  # a, b, c, d, e, f, g
nb_zones = 2   # mid, end

for i, val in enumerate(popt_sd):
    # Affichage de la valeur avec virgule décimale
    print(str(val).replace('.', ','))

    # Ligne vide entre zones (toutes les 7 valeurs)
    if (i + 1) % nb_params == 0 and (i + 1) < nb_params * nb_zones:
        print("")

# --- Affichage graphique ---
def plot_fit_result_onephase(w, x, y, z, n_discs, params):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    phase = 64 - n_discs
    x_phase = []
    y_phase = []
    z_phase = []
    for ww, xx, yy, zz in zip(w, x, y, z):
        if ww == n_discs:
            x_phase.append(xx)
            y_phase.append(yy)
            z_phase.append(zz)
    ax.plot(x_phase, y_phase, z_phase, ms=3, marker="o", linestyle='None', label=f"nb empty {phase}")
    mx, my = np.meshgrid(range(phase), range(phase))
    # Aplatir pour passer au modèle
    mx_flat = mx.ravel()
    my_flat = my.ravel()
    mz_flat = f_max((np.full_like(mx_flat, n_discs), mx_flat, my_flat), *params)

    # Remodeler à la taille originale
    mz = mz_flat.reshape(mx.shape)

    # Tracer la surface
    ax.plot_wireframe(mx, my, mz, rstride=4, cstride=2)
    
    ax.set_xlabel('shallow depth')
    ax.set_ylabel('depth')
    ax.set_zlabel('error')
    ax.set_xlim((0, phase))
    ax.set_ylim((0, phase))
    ax.set_zlim((0, phase))
    ax.legend()
    plt.show()


# --- Visualisation sur plusieurs phases ---
for i in [14, 24, 34, 44]:
    plot_fit_result_onephase(w_n_discs_sd, x_depth1_sd, y_depth2_sd, z_sd, i, popt_sd)

'''
constexpr double probcut_a[] = {-0.00906229, 0.00483133};
constexpr double probcut_b[] = {0.12115910, 0.07975382};
constexpr double probcut_c[] = {-0.12071974, -0.01763140};
constexpr double probcut_d[] = {-0.47638522, -3.58586370};
constexpr double probcut_e[] = {-2.93341676, 9.99659776};
constexpr double probcut_f[] = {-5.44314615, -9.52512272};
constexpr double probcut_g[] = {0.01672187, 6.76910942};
'''

