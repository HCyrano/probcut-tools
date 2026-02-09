import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

nb_params = 7  # 6 = zone_2D, 7 = zone_3D



# --- Chargement des données (inchangé) ---
data_files = ['../data/midgame/probcut_mid.txt'] #, '../data/midgame/probcut_extrapolated.txt'
data_files_end = ['../data/endgame/probcut_end.txt', '../data/endgame/probcut_end_ext.txt']

data = [[[[] for _ in range(61)] for _ in range(61)] for _ in range(65)]

for data_file in data_files:
    try:
        with open(data_file, 'r') as f:
            raw_data = f.read().splitlines()
        for datum in raw_data:
            n_discs, depth1, depth2, error = [int(elem) for elem in datum.split()]
            if n_discs == 26 and depth1 == 10 and depth2 == 24:
                print(error)

            if n_discs > 5 and depth2 > 1:
                data[n_discs][depth1][depth2].append(error)
                
    except (FileNotFoundError, IOError) as e:
        print(f'cannot open {data_file}: {e}')
    
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

for n_discs in range(8, len(data)):
    for depth1 in range(len(data[n_discs])):
        for depth2 in range(len(data[n_discs][depth1])):
            if len(data[n_discs][depth1][depth2]) >= 10:

                sd = statistics.stdev(data[n_discs][depth1][depth2])

                w_n_discs_sd.append(n_discs)
                x_depth1_sd.append(depth1)
                y_depth2_sd.append(depth2)
                z_sd.append(sd)

                weight_sd.append(1)
                


# --- Ajout des points de guidage ---

for n_discs in range(8, 64):
    depth2 = 30 + (n_discs & 1)
    depth1 = (n_discs & 1)
    w_n_discs_sd.append(n_discs)
    x_depth1_sd.append(depth1)
    y_depth2_sd.append(depth2)
    z_sd.append(11)
    weight_sd.append(0.1)

'''

for n_discs in range(8, 25):
    depth2 = 40 + (n_discs & 1)
    depth1 = 0 + (n_discs & 1)
    w_n_discs_sd.append(n_discs)
    x_depth1_sd.append(depth1)
    y_depth2_sd.append(depth2)
    z_sd.append(16.0)
    weight_sd.append(0.20)
    
'''
# remplace la stddev par 2,5 pour depth1 == depth2-2
for i in range(len(z_sd)):
    if x_depth1_sd[i] == y_depth2_sd[i] - 2:
        # Remplacer par une nouvelle valeur
        z_sd[i] = 2.5
        weight_sd[i] = 1  # ou un poids réduit

'''        
for n_discs in range(8, 63):
    for depth in (0, 64-n_discs+1):
        w_n_discs_sd.append(n_discs)
        x_depth1_sd.append(depth)
        y_depth2_sd.append(depth)
        z_sd.append(0)
        weight_sd.append(0,95)
'''


# polynome 3D - 1 zone
def f_single_3D(wxy, a, b, c, d, e, f, g):
    w, x, y = wxy
    w = 64 - np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    r = a*w + b*x + c*y
    return d*r**3 + e*r**2 + f*r + g


# polynome 2D - 1 zone
def f_single_2D(wxy, a, b, c, d, e, f):
    w, x, y = wxy
    w = 64 - np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    r = a*w + b*x + c*y
    return d*r**2 + e*r + f


# --- Ajustement global avec une seule zone ---
# Sélectionner automatiquement la fonction selon nb_params
f_model = f_single_3D if nb_params == 7 else f_single_2D

# --- bornage de la fonction ---
def f_max(wxy, *params):
    return np.minimum(40.0, np.maximum(0.0, f_model(wxy, *params)))



p0 = np.ones(nb_params)
popt_sd, pcov_sd = curve_fit(
    f_model,
    (w_n_discs_sd, x_depth1_sd, y_depth2_sd),
    z_sd,
    p0,
    sigma=weight_sd,
    absolute_sigma=True,
    maxfev=10000
)

# ============================================================================
# AJOUT : MÉTRIQUES DE PERFORMANCE DU MODÈLE
# ============================================================================

print("\n" + "="*80)
print("MÉTRIQUES DE PERFORMANCE DU MODÈLE (1 ZONE)")
print("="*80 + "\n")

# --- Prédictions du modèle ---
z_pred = f_model((w_n_discs_sd, x_depth1_sd, y_depth2_sd), *popt_sd)
z_pred_bounded = f_max((w_n_discs_sd, x_depth1_sd, y_depth2_sd), *popt_sd)

# --- Conversion en numpy arrays ---
z_true = np.array(z_sd)
z_pred = np.array(z_pred)
z_pred_bounded = np.array(z_pred_bounded)
weights = np.array(weight_sd)

# --- Métriques globales ---
print("📊 MÉTRIQUES GLOBALES (tous les points)")
print("-" * 80)
r2 = r2_score(z_true, z_pred_bounded)
rmse = np.sqrt(mean_squared_error(z_true, z_pred_bounded))
mae = mean_absolute_error(z_true, z_pred_bounded)
max_error = np.max(np.abs(z_true - z_pred_bounded))

print(f"R² Score                : {r2:.6f}")
print(f"RMSE (Root Mean Sq Err) : {rmse:.4f}")
print(f"MAE (Mean Absolute Err) : {mae:.4f}")
print(f"Erreur maximale         : {max_error:.4f}")
print()

# --- Métriques par type de point ---
print("📊 MÉTRIQUES PAR TYPE DE POINT")
print("-" * 80)

# Séparer les points réels des points de guidage
# Points avec poids = 1 sont les données réelles
real_data_mask = weights == 1.0
guide_data_mask = weights != 1.0

if np.any(real_data_mask):
    r2_real = r2_score(z_true[real_data_mask], z_pred_bounded[real_data_mask])
    rmse_real = np.sqrt(mean_squared_error(z_true[real_data_mask], z_pred_bounded[real_data_mask]))
    mae_real = mean_absolute_error(z_true[real_data_mask], z_pred_bounded[real_data_mask])
    print(f"Points de données réelles ({np.sum(real_data_mask)} points):")
    print(f"  R²   : {r2_real:.6f}")
    print(f"  RMSE : {rmse_real:.4f}")
    print(f"  MAE  : {mae_real:.4f}")
    print()

if np.any(guide_data_mask):
    r2_guide = r2_score(z_true[guide_data_mask], z_pred_bounded[guide_data_mask])
    rmse_guide = np.sqrt(mean_squared_error(z_true[guide_data_mask], z_pred_bounded[guide_data_mask]))
    mae_guide = mean_absolute_error(z_true[guide_data_mask], z_pred_bounded[guide_data_mask])
    print(f"Points de guidage ({np.sum(guide_data_mask)} points):")
    print(f"  R²   : {r2_guide:.6f}")
    print(f"  RMSE : {rmse_guide:.4f}")
    print(f"  MAE  : {mae_guide:.4f}")
    print()

# --- Métriques par phase de jeu ---
print("📊 MÉTRIQUES PAR PHASE DE JEU (données réelles)")
print("-" * 80)

w_array = np.array(w_n_discs_sd)
phases = [
    ("Early game (6-20 disques)", 6, 20),
    ("Mid game (21-40 disques)", 21, 40),
    ("Late game (41-55 disques)", 41, 55),
    ("End game (56-64 disques)", 56, 64)
]

for phase_name, n_min, n_max in phases:
    phase_mask = (w_array >= n_min) & (w_array <= n_max) & real_data_mask
    if np.sum(phase_mask) > 10:  # Au moins 10 points
        r2_phase = r2_score(z_true[phase_mask], z_pred_bounded[phase_mask])
        rmse_phase = np.sqrt(mean_squared_error(z_true[phase_mask], z_pred_bounded[phase_mask]))
        mae_phase = mean_absolute_error(z_true[phase_mask], z_pred_bounded[phase_mask])
        print(f"{phase_name}:")
        print(f"  Nombre de points : {np.sum(phase_mask)}")
        print(f"  R²               : {r2_phase:.6f}")
        print(f"  RMSE             : {rmse_phase:.4f}")
        print(f"  MAE              : {mae_phase:.4f}")
        print()

# --- Distribution des erreurs ---
print("📊 DISTRIBUTION DES ERREURS (points réels)")
print("-" * 80)

residuals = z_true[real_data_mask] - z_pred_bounded[real_data_mask]
print(f"Moyenne des résidus     : {np.mean(residuals):.4f} (devrait être proche de 0)")
print(f"Écart-type des résidus  : {np.std(residuals):.4f}")
print(f"Médiane des résidus     : {np.median(residuals):.4f}")
print()

# Percentiles
percentiles = [50, 75, 90, 95, 99]
abs_residuals = np.abs(residuals)
print("Erreur absolue par percentile:")
for p in percentiles:
    val = np.percentile(abs_residuals, p)
    print(f"  {p}% des prédictions ont une erreur < {val:.4f}")
print()

# --- Vérification du surapprentissage ---
print("📊 ANALYSE DU SURAPPRENTISSAGE")
print("-" * 80)
n_params = nb_params
n_samples_real = np.sum(real_data_mask)
ratio = n_samples_real / n_params

print(f"Nombre de paramètres    : {n_params}")
print(f"Nombre de points réels  : {n_samples_real}")
print(f"Ratio échantillons/params: {ratio:.1f}")

if ratio < 10:
    print("  ⚠️  ATTENTION: Ratio faible, risque de surapprentissage élevé")
elif ratio < 30:
    print("  ⚠️  Ratio acceptable mais surveillez la validation croisée")
else:
    print("  ✓  Ratio sain")
print()

# --- Statistiques sur les valeurs prédites ---
print("📊 STATISTIQUES DES VALEURS PRÉDITES")
print("-" * 80)
print(f"Min prédit (non borné)  : {np.min(z_pred):.4f}")
print(f"Max prédit (non borné)  : {np.max(z_pred):.4f}")
print(f"% de valeurs hors [0,40]: {100 * np.sum((z_pred < 0) | (z_pred > 40)) / len(z_pred):.2f}%")
print()

print("="*80)
print("FIN DES MÉTRIQUES")
print("="*80 + "\n")

# ============================================================================
# FIN DES MÉTRIQUES
# ============================================================================

# --- Sortie format C++ : probcut_a = value; etc. ---

# Déterminer automatiquement les noms de paramètres
param_names = ["a", "b", "c", "d", "e", "f", "g"][:nb_params]

# Pour 1 zone, on a juste 1 valeur par paramètre
for j, name in enumerate(param_names):
    val = popt_sd[j]
    print(f"constexpr double probcut_{name} = {val:.8f};")

print()

# --- Affichage copiable dans un tableur : une valeur par ligne ---

for i, val in enumerate(popt_sd):
    # Affichage de la valeur avec virgule décimale
    print(str(val).replace('.', ','))

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

# --- Graphiques d'analyse des résidus ---
print("\nGénération des graphiques d'analyse des résidus...")

# Préparer les données pour les graphiques
z_true_all = np.array(z_sd)
z_pred_all = f_max((w_n_discs_sd, x_depth1_sd, y_depth2_sd), *popt_sd)
weights_all = np.array(weight_sd)
real_mask = weights_all == 1.0

z_true_real = z_true_all[real_mask]
z_pred_real = z_pred_all[real_mask]
residuals_real = z_true_real - z_pred_real

# Figure 1: Scatter plot prédictions vs vraies valeurs
fig1, axes = plt.subplots(2, 2, figsize=(14, 12))

# Subplot 1: Prédictions vs Valeurs réelles
axes[0, 0].scatter(z_true_real, z_pred_real, alpha=0.5, s=10)
axes[0, 0].plot([0, max(z_true_real)], [0, max(z_true_real)], 'r--', label='Prédiction parfaite')
axes[0, 0].set_xlabel('Valeurs réelles')
axes[0, 0].set_ylabel('Valeurs prédites')
axes[0, 0].set_title('Prédictions vs Valeurs réelles (données réelles) - 1 ZONE')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Résidus vs Valeurs prédites
axes[0, 1].scatter(z_pred_real, residuals_real, alpha=0.5, s=10)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Valeurs prédites')
axes[0, 1].set_ylabel('Résidus (réel - prédit)')
axes[0, 1].set_title('Résidus vs Valeurs prédites')
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Histogramme des résidus
axes[1, 0].hist(residuals_real, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', label='Résidu = 0')
axes[1, 0].set_xlabel('Résidus')
axes[1, 0].set_ylabel('Fréquence')
axes[1, 0].set_title(f'Distribution des résidus (μ={np.mean(residuals_real):.3f}, σ={np.std(residuals_real):.3f})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Q-Q plot (normalité des résidus)
from scipy import stats
stats.probplot(residuals_real, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (test de normalité des résidus)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_analysis_1zone.png', dpi=150, bbox_inches='tight')
print("✓ Graphique des résidus sauvegardé: residuals_analysis_1zone.png")

# Figure 2: Analyse par phase de jeu
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

w_array_real = np.array([w_n_discs_sd[i] for i, mask in enumerate(real_mask) if mask])
phases_plot = [
    ("Early (6-20)", 6, 20, axes2[0, 0]),
    ("Mid (21-40)", 21, 40, axes2[0, 1]),
    ("Late (41-55)", 41, 55, axes2[1, 0]),
    ("End (56-64)", 56, 64, axes2[1, 1])
]

for phase_name, n_min, n_max, ax in phases_plot:
    phase_mask = (w_array_real >= n_min) & (w_array_real <= n_max)
    if np.sum(phase_mask) > 0:
        ax.scatter(z_true_real[phase_mask], z_pred_real[phase_mask], alpha=0.6, s=15)
        max_val = max(np.max(z_true_real[phase_mask]), np.max(z_pred_real[phase_mask]))
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
        ax.set_xlabel('Valeurs réelles')
        ax.set_ylabel('Valeurs prédites')
        ax.set_title(f'{phase_name} - n={np.sum(phase_mask)}')
        ax.grid(True, alpha=0.3)
        
        # Ajouter R² sur le graphique
        if np.sum(phase_mask) > 1:
            r2_phase = r2_score(z_true_real[phase_mask], z_pred_real[phase_mask])
            ax.text(0.05, 0.95, f'R²={r2_phase:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('phase_analysis_1zone.png', dpi=150, bbox_inches='tight')
print("✓ Graphique par phase sauvegardé: phase_analysis_1zone.png")

plt.show()
