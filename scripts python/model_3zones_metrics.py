import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# CHOISIR ICI L'APPROCHE À TESTER (1 ou 2)
# 1: Approche w_mid = 1 - w_early - w_end
# 2: Approche Gaussiennes
selected_approach = 2  

# ============================================================================
# CONFIGURATION DES POINTS DE GUIDAGE
# ============================================================================
# Options pour les points de guidage :
# 0 = Pas de points de guidage (recommandé pour tester)
# 1 = Points de guidage légers (poids 0.1-0.3)
# 2 = Points de guidage modérés (poids 0.5)
# 3 = Points de guidage originaux (poids 0.95)
GUIDANCE_MODE = 2  # ← CHANGEZ CETTE VALEUR (0, 1, 2, ou 3)

# --- Graphiques (optionnel) ---
SHOW_PLOTS = True  # Mettre à True pour afficher les graphiques


nb_params = 7  # 6 = zone_2D, 7 = zone_3D
nb_zones = 3 




print("\n" + "="*80)
print(f"MODE DE GUIDAGE: {GUIDANCE_MODE}")
if GUIDANCE_MODE == 0:
    print("Pas de points de guidage - optimisation basée uniquement sur les données réelles")
elif GUIDANCE_MODE == 1:
    print("Points de guidage légers - influence minimale sur l'optimisation")
elif GUIDANCE_MODE == 2:
    print("Points de guidage modérés - influence équilibrée")
else:
    print("Points de guidage originaux - forte influence (peut causer des échecs)")
print("="*80 + "\n")

# ============================================================================
# CHARGEMENT DES DONEES & CALCUL DES STD DEV
# ============================================================================

print("\n" + "="*80)
print("Lecture DATA et calcul des std dev...")
print("="*80 + "\n")


data_files = ['../data/midgame/probcut_mid.txt', '../data/midgame/probcut_mid1.txt'] 
data_files_end = ['../data/endgame/probcut_end.txt', '../data/endgame/probcut_end_ext.txt']

data = [[[[] for _ in range(61)] for _ in range(61)] for _ in range(65)]

for data_file in data_files:
    try:
        with open(data_file, 'r') as f:
            print(f'read {data_file}')
            raw_data = f.read().splitlines()
        for datum in raw_data:
            n_discs, depth1, depth2, error = [int(elem) for elem in datum.split()]
            if n_discs > 5 and depth2 > 1:
                data[n_discs][depth1][depth2].append(error)
    except (FileNotFoundError, IOError) as e:
        print(f'cannot open {data_file}: {e}')
    
for data_file in data_files_end:
    try:
        with open(data_file, 'r') as f:
            print(f'read {data_file}')
            raw_data = f.read().splitlines()
        for datum in raw_data:
            n_discs, depth1, error = [int(elem) for elem in datum.split()]
            depth2 = 64 - n_discs
            data[n_discs][depth1][depth2].append(error)
    except (FileNotFoundError, IOError) as e:
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

print(f"\nNombre de points de données réelles: {len([w for w in weight_sd if w == 1])}")


# ============================================================================
# --- Ajout des points de guidage selon le mode choisi ---
# ============================================================================

if GUIDANCE_MODE > 0:
    # Déterminer le poids selon le mode
    if GUIDANCE_MODE == 1:
        guidance_weight = 0.2
    elif GUIDANCE_MODE == 2:
        guidance_weight = 0.5
    else:  # GUIDANCE_MODE == 3
        guidance_weight = 0.95
    
    # Ajout de points de guidage stratégiques PAR ZONE
    guidance_count = 0
    
    # Zone EARLY (10-20 disques, 44-54 cases vides)
    for n_discs in range(10, 20):
        depth2 = min(30, 64 - n_discs - 2)
        depth1 = (n_discs & 1)
        w_n_discs_sd.append(n_discs)
        x_depth1_sd.append(depth1)
        y_depth2_sd.append(depth2)   
        z_sd.append(12)  # sd cible pour early game
        weight_sd.append(guidance_weight)
        guidance_count += 1
    
    # Zone MID (21-35 disques, 29-43 cases vides)
    for n_discs in range(21, 35):
        depth2 = min(25, 64 - n_discs - 2)
        depth1 = (n_discs & 1)
        w_n_discs_sd.append(n_discs)
        x_depth1_sd.append(depth1)
        y_depth2_sd.append(depth2)   
        z_sd.append(10)  # sd cible pour mid game
        weight_sd.append(guidance_weight)
        guidance_count += 1
    
    # Zone LATE (40-50 disques, 14-24 cases vides)
    for n_discs in range(40, 50):
        depth2 = min(20, 64 - n_discs - 2)
        depth1 = (n_discs & 1)
        w_n_discs_sd.append(n_discs)
        x_depth1_sd.append(depth1)
        y_depth2_sd.append(depth2)   
        z_sd.append(8)  # sd cible pour late game
        weight_sd.append(guidance_weight)
        guidance_count += 1
    
    print(f"Nombre de points de guidage ajoutés: {guidance_count}")
    print(f"Poids des points de guidage: {guidance_weight}")
else:
    print("Aucun point de guidage ajouté")

print(f"Nombre total de points: {len(weight_sd)}\n")

# ============================================================================
# fonctions utilitaires
# ============================================================================

# --- Fonction sigmoïde ---
def sigmoid(x, x0, k=0.5):
    """Fonction sigmoïde utilisée pour adoucir les transitions."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def poly_model(a,b,c,d,e,f,g, w, x, y):
    r = a*w + b*x + c*y
    return d*r**3 + e*r**2 + f*r + g

# --- bornage de la fonction ---
def f_max(wxy, *params):
    return np.minimum(40.0, np.maximum(0.0, f_model(wxy, *params)))


# ============================================================================
# APPROCHE 1 : Formulation w_mid = 1 - w_early - w_end
# ============================================================================
def f_zoned_3D_approach1(wxy,
            a1,b1,c1,d1,e1,f1,g1,
            a2,b2,c2,d2,e2,f2,g2,
            a3,b3,c3,d3,e3,f3,g3
    ):
    w, x, y = wxy
    w = 64 - np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Correction des seuils pour un pic central entre 18 et 42
    s1 = sigmoid(w, 18, k=0.4)
    s2 = sigmoid(w, 42, k=0.4)
    
    w_early = (1 - s1)
    w_end = s2
    w_mid = np.maximum(0.0, 1.0 - w_early - w_end)

    result_early = poly_model(a1,b1,c1,d1,e1,f1,g1, w, x, y)
    result_mid = poly_model(a2,b2,c2,d2,e2,f2,g2, w, x, y)
    result_end = poly_model(a3,b3,c3,d3,e3,f3,g3, w, x, y)

    return w_early*result_early + w_mid*result_mid + w_end*result_end

# ============================================================================
# APPROCHE 2 : Cloches gaussiennes
# ============================================================================
def f_zoned_3D_approach2(wxy,
            a1,b1,c1,d1,e1,f1,g1,
            a2,b2,c2,d2,e2,f2,g2,
            a3,b3,c3,d3,e3,f3,g3
    ):
    w, x, y = wxy
    w = 64 - np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    sigma = 12
    w_early_raw = np.exp(-((w - 50)**2) / (2 * sigma**2))
    w_mid_raw = np.exp(-((w - 30)**2) / (2 * sigma**2))
    w_end_raw = np.exp(-((w - 10)**2) / (2 * sigma**2))
    
    total = w_early_raw + w_mid_raw + w_end_raw
    w_early = w_early_raw / total
    w_mid = w_mid_raw / total
    w_end = w_end_raw / total

    result_early = poly_model(a1,b1,c1,d1,e1,f1,g1, w, x, y)
    result_mid = poly_model(a2,b2,c2,d2,e2,f2,g2, w, x, y)
    result_end = poly_model(a3,b3,c3,d3,e3,f3,g3, w, x, y)

    return w_early*result_early + w_mid*result_mid + w_end*result_end


# ============================================================================
# FONCTION DE VISUALISATION DES POIDS
# ============================================================================
def plot_zone_weights():
    w_values = np.linspace(5, 60, 200)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    approaches = [
        ("Approche 1: Somme pondérée (1 - s1 - s2)", f_zoned_3D_approach1),
        ("Approche 2: Cloches gaussiennes", f_zoned_3D_approach2)
    ]
    
    for idx, (title, func) in enumerate(approaches):
        ax = axes[idx]
        weights_e, weights_m, weights_en = [], [], []
        
        for w_val in w_values:
            w_arr = np.array([w_val])
            if func == f_zoned_3D_approach1:
                s1 = sigmoid(w_arr, 18, k=0.4)[0]
                s2 = sigmoid(w_arr, 42, k=0.4)[0]
                we = 1.0 - s1
                wen = s2
                wm = max(0.0, 1.0 - we - wen)
            else: # gauss
                sigma = 12
                we_r = np.exp(-((w_arr - 50)**2) / (2 * sigma**2))[0]
                wm_r = np.exp(-((w_arr - 30)**2) / (2 * sigma**2))[0]
                wen_r = np.exp(-((w_arr - 10)**2) / (2 * sigma**2))[0]
                t = we_r + wm_r + wen_r
                we, wm, wen = we_r/t, wm_r/t, wen_r/t
            
            weights_e.append(we); weights_m.append(wm); weights_en.append(wen)
        
        n_discs = 64 - w_values
        ax.plot(n_discs, weights_e, 'b-', label='Early')
        ax.plot(n_discs, weights_m, 'g-', label='Mid')
        ax.plot(n_discs, weights_en, 'r-', label='End')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zone_weights_comparison_v2.png')
    return fig

# ============================================================================
# SÉLECTION DE L'APPROCHE
# ============================================================================
print("="*80)
print("SÉLECTION DE L'APPROCHE")
print("="*80 + "\n")


# --- Dictionnaire de sélection mis à jour ---
approaches_dict = {
    1: ("Approche 1: Logique soustractive", f_zoned_3D_approach1),
    2: ("Approche 2: Cloches gaussiennes", f_zoned_3D_approach2)
}

approach_name, f_model = approaches_dict[selected_approach]

print(f"🎯 APPROCHE SÉLECTIONNÉE: {approach_name}\n")
print("="*80 + "\n")


# Visualisation des poids
# plot_zone_weights()


# ============================================================================
# OPTIMISATION
# ============================================================================
print("Démarrage de l'optimisation avec curve_fit...")
print(f"Nombre de paramètres à optimiser: {nb_params * nb_zones}")
print()

try:
    p0 = np.ones(nb_params * nb_zones)
    popt_sd, pcov_sd = curve_fit(
        f_model,
        (w_n_discs_sd, x_depth1_sd, y_depth2_sd),
        z_sd,
        p0,
        sigma=weight_sd,
        absolute_sigma=True,
        maxfev=20000  # Augmenté pour plus de chances de convergence
    )
    print("✓ Optimisation réussie!\n")
    
except RuntimeError as e:
    print(f"❌ ÉCHEC DE L'OPTIMISATION: {e}")
    print("\nSuggestions:")
    print("  1. Réduire GUIDANCE_MODE")
    print("  2. Essayer une autre approche")
    print("  3. Vérifier que les fichiers de données sont présents")
    print()
    import sys
    sys.exit(1)

# ============================================================================
# MÉTRIQUES DE PERFORMANCE
# ============================================================================

print("\n" + "="*80)
print(f"MÉTRIQUES DE PERFORMANCE - {approach_name}")
print("="*80 + "\n")

z_pred = f_model((w_n_discs_sd, x_depth1_sd, y_depth2_sd), *popt_sd)
z_pred_bounded = f_max((w_n_discs_sd, x_depth1_sd, y_depth2_sd), *popt_sd)

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

print(f"R² (coefficient de détermination) : {r2:.6f}")
print(f"RMSE (erreur quadratique moyenne)  : {rmse:.4f}")
print(f"MAE (erreur absolue moyenne)       : {mae:.4f}")
print()

# --- Séparer données réelles vs guidage ---
real_data_mask = weights == 1.0
guide_data_mask = weights < 1.0

if np.sum(real_data_mask) > 0:
    r2_real = r2_score(z_true[real_data_mask], z_pred_bounded[real_data_mask])
    rmse_real = np.sqrt(mean_squared_error(z_true[real_data_mask], z_pred_bounded[real_data_mask]))
    mae_real = mean_absolute_error(z_true[real_data_mask], z_pred_bounded[real_data_mask])
    
    print(f"Données réelles uniquement ({np.sum(real_data_mask)} points):")
    print(f"  R²   : {r2_real:.6f}")
    print(f"  RMSE : {rmse_real:.4f}")
    print(f"  MAE  : {mae_real:.4f}")
    print()

if np.sum(guide_data_mask) > 0:
    r2_guide = r2_score(z_true[guide_data_mask], z_pred_bounded[guide_data_mask])
    rmse_guide = np.sqrt(mean_squared_error(z_true[guide_data_mask], z_pred_bounded[guide_data_mask]))
    mae_guide = mean_absolute_error(z_true[guide_data_mask], z_pred_bounded[guide_data_mask])
    
    print(f"Points de guidage uniquement ({np.sum(guide_data_mask)} points):")
    print(f"  R²   : {r2_guide:.6f}")
    print(f"  RMSE : {rmse_guide:.4f}")
    print(f"  MAE  : {mae_guide:.4f}")
    print()

# --- Métriques par phase de jeu ---
print("📊 MÉTRIQUES PAR PHASE DE JEU")
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
    if np.sum(phase_mask) > 10:
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
print(f"Moyenne des résidus     : {np.mean(residuals):.4f}")
print(f"Écart-type des résidus  : {np.std(residuals):.4f}")
print(f"Médiane des résidus     : {np.median(residuals):.4f}")
print()

percentiles = [50, 75, 90, 95, 99]
abs_residuals = np.abs(residuals)
print("Erreur absolue par percentile:")
for p in percentiles:
    val = np.percentile(abs_residuals, p)
    print(f"  {p}% des prédictions ont une erreur < {val:.4f}")
print()

# --- Analyse du surapprentissage ---
print("📊 ANALYSE DU SURAPPRENTISSAGE")
print("-" * 80)
n_params = nb_params * nb_zones
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

# --- Sortie format C++ ---
n_params_per_zone = len(popt_sd) // nb_zones
param_names = ["a", "b", "c", "d", "e", "f", "g"][:n_params_per_zone]

print("\n" + "="*80)
print("PARAMÈTRES OPTIMISÉS (FORMAT C++)")
print("="*80 + "\n")

for j, name in enumerate(param_names):
    vals = [popt_sd[j + nb_params*i] for i in range(nb_zones)]
    vals_str = ", ".join(f"{v:.8f}" for v in vals)
    print(f"constexpr double probcut_{name}[] = {{{vals_str}}};")

print()

# Affichage tableur
print("="*80)
print("PARAMÈTRES (FORMAT TABLEUR)")
print("="*80 + "\n")

for i, val in enumerate(popt_sd):
    print(str(val).replace('.', ','))
    if (i + 1) % nb_params == 0 and (i + 1) < nb_params * nb_zones:
        print("")

# ============================================================================
# GÉNÉRATION DES GRAPHIQUES D'ANALYSE (RÉSIDUS ET PHASES)
# ============================================================================
print("\n" + "="*80)
print("Génération des graphiques d'analyse statistique...")
print("="*80 + "\n")


# Préparation des données pour les graphiques (uniquement points réels)
z_true_real = z_true[real_data_mask]
z_pred_real = z_pred_bounded[real_data_mask]
residuals_real = z_true_real - z_pred_real
w_array_real = w_array[real_data_mask]

# --- Figure 1: Analyse des résidus ---
fig_res, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Prédictions vs Valeurs réelles
axes[0, 0].scatter(z_true_real, z_pred_real, alpha=0.5, s=10, color='#1f77b4')
max_val = max(z_true_real.max(), z_pred_real.max())
axes[0, 0].plot([0, max_val], [0, max_val], 'r--', label='Prédiction parfaite')
axes[0, 0].set_xlabel('Valeurs réelles (SD)')
axes[0, 0].set_ylabel('Valeurs prédites')
axes[0, 0].set_title(f'Prédictions vs Réalité - {approach_name}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Résidus vs Valeurs prédites
axes[0, 1].scatter(z_pred_real, residuals_real, alpha=0.5, s=10, color='#ff7f0e')
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Valeurs prédites')
axes[0, 1].set_ylabel('Résidus (Réel - Prédit)')
axes[0, 1].set_title('Analyse de l\'homoscédasticité')
axes[0, 1].grid(True, alpha=0.3)

# 3. Histogramme des résidus
axes[1, 0].hist(residuals_real, bins=50, edgecolor='black', alpha=0.7, color='#2ca02c')
axes[1, 0].axvline(x=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Erreur (Résidus)')
axes[1, 0].set_ylabel('Fréquence')
axes[1, 0].set_title(f'Distribution des erreurs (μ={np.mean(residuals_real):.3f})')
axes[1, 0].grid(True, alpha=0.3)

# 4. Q-Q Plot
from scipy import stats
stats.probplot(residuals_real, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normalité des résidus)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'residuals_analysis_{selected_approach}.png', dpi=150)
print(f"✓ Graphique des résidus sauvegardé: residuals_analysis_{selected_approach}.png")

# --- Figure 2: Analyse par phase de jeu ---
fig_phase, axes2 = plt.subplots(2, 2, figsize=(14, 10))
plt.suptitle(f"Performance par phase - {approach_name}", fontsize=16)

phases_plot = [
    ("Early (6-20)", 6, 20, axes2[0, 0]),
    ("Mid (21-40)", 21, 40, axes2[0, 1]),
    ("Late (41-55)", 41, 55, axes2[1, 0]),
    ("End (56-64)", 56, 64, axes2[1, 1])
]

for phase_title, n_min, n_max, ax in phases_plot:
    phase_mask = (w_array_real >= n_min) & (w_array_real <= n_max)
    if np.sum(phase_mask) > 1:
        z_t_ph = z_true_real[phase_mask]
        z_p_ph = z_pred_real[phase_mask]
        ax.scatter(z_t_ph, z_p_ph, alpha=0.6, s=15)
        m_val = max(z_t_ph.max(), z_p_ph.max())
        ax.plot([0, m_val], [0, m_val], 'r--', alpha=0.7)
        
        r2_ph = r2_score(z_t_ph, z_p_ph)
        ax.text(0.05, 0.92, f'R²={r2_ph:.4f}\nn={np.sum(phase_mask)}', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_title(phase_title)
        ax.set_xlabel('Réel')
        ax.set_ylabel('Prédit')
        ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'phase_analysis_{selected_approach}.png', dpi=150)
print(f"✓ Graphique par phase sauvegardé: phase_analysis_{selected_approach}.png")

# ============================================================================
# GÉNÉRATION DES GRAPHIQUES PAR STAGES
# ============================================================================
print("\nGénération des graphiques par stages..")


if SHOW_PLOTS:
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
        mx_flat = mx.ravel()
        my_flat = my.ravel()
        mz_flat = f_max((np.full_like(mx_flat, n_discs), mx_flat, my_flat), *params)
        mz = mz_flat.reshape(mx.shape)
        ax.plot_wireframe(mx, my, mz, rstride=4, cstride=2)
        ax.set_xlabel('shallow depth')
        ax.set_ylabel('depth')
        ax.set_zlabel('error')
        ax.set_xlim((0, phase))
        ax.set_ylim((0, phase))
        ax.set_zlim((0, phase))
        ax.legend()
        plt.show()

    for i in [14, 24, 34, 44]:
        plot_fit_result_onephase(w_n_discs_sd, x_depth1_sd, y_depth2_sd, z_sd, i, popt_sd)

print("\n" + "="*80)
print("✅ ANALYSE TERMINÉE")
print("="*80)
print(f"\nConfiguration utilisée:")
print(f"  - Approche: {selected_approach}")
print(f"  - Mode de guidage: {GUIDANCE_MODE}")
#print(f"  - Fichier de sortie: zone_weights_comparison.png")
