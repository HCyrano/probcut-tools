import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import animation
import math

#depth1: short
#depth2: long

data_files = ['../data/midgame/probcut_mid.txt', '../data/midgame/probcut_mid_ext.txt']
data_files_end = ['../data/endgame/probcut_end.txt', '../data/endgame/probcut_end_ext.txt', '../data/endgame/probcut_end_ext2.txt']

#on definit un tableau data[65][61][61][];
data = [[[[] for _ in range(61)] for _ in range(61)] for _ in range(65)] # n_discs, depth1, depth2 (depth1 < depth2)


# chargement des data
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

# --- Nouvelle fonction : zones internes ---
def sigmoid(x, x0, k=0.5):
    """Fonction sigmoïde utilisée pour adoucir les transitions."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


# definition des tableaux de resultats
# pour la deviation standart
w_n_discs_sd = []
x_depth1_sd = []
y_depth2_sd = []
z_sd = []
weight_sd = []

'''
# pour la moyenne
w_n_discs_mean = []
x_depth1_mean = []
y_depth2_mean = []
z_mean = []
weight_mean = []
'''


# calcul la moyenne et la deviation standart
# par n_discs, shallow_depth, et depth
for n_discs in range(6, len(data)):
    for depth1 in range(len(data[n_discs])): # short
        for depth2 in range(len(data[n_discs][depth1])): # long
            if len(data[n_discs][depth1][depth2]) >= 10:
                
                #initialisation des valeurs
                
                #calcul de la moyenne (force a 0)
                #mean = statistics.mean(data[n_discs][depth1][depth2])
                
                #idem pour la moyenne (la moyenne etant 0.0 pas vraiment utile)
                #w_n_discs_mean.append(n_discs)
                #x_depth1_mean.append(depth1)
                #y_depth2_mean.append(depth2)
                #z_mean.append(mean)
                #weight_mean.append(1)

                
                #calcul de la deviation standart (ecart type)
                sd = statistics.stdev(data[n_discs][depth1][depth2])
                                
                #remplisage des tableaux
                w_n_discs_sd.append(n_discs)
                x_depth1_sd.append(depth1)
                y_depth2_sd.append(depth2)
                z_sd.append(sd)
                # Le poids lié à l’écart-type : En statistique (ex. en physique expérimentale), on donne plus de poids aux mesures plus précises.
                # La règle la plus courante est : poids w(i) = 1/stdev(i)**2
                weight_sd.append(0.1)
                
                

# placement manuel des points pour guider le modele

# a) midgame at depth 31/1 or 30/0
for n_discs in range(6, 34):
    depth2 = 30 + (n_discs & 1)
    depth1 = 0 + (n_discs & 1)
    
    w_n_discs_sd.append(n_discs)
    x_depth1_sd.append(depth1)
    y_depth2_sd.append(depth2)
    
    z_sd.append(12.0 - 0.143 * ((n_discs+(n_discs & 1))-6))
    weight_sd.append(0.1) # 1/10

# b) midgame+endgame at depth 41/1 or 40/0
for n_discs in range(6, 25):
    depth2 = 40 + (n_discs & 1)
    depth1 = 0 + (n_discs & 1)
    
    w_n_discs_sd.append(n_discs)
    x_depth1_sd.append(depth1)
    y_depth2_sd.append(depth2)
    
    z_sd.append(18.0 - 0.111 * ((n_discs+(n_discs & 1))-6))
    weight_sd.append(0.025)

# c) midgame+endgame at depth vs depth-2 => stdev = 3
for n_discs in range(6, 63):
    for depth2 in range(2, 64-n_discs+1):
        depth1 = depth2-2
    
        w_n_discs_sd.append(n_discs)
        x_depth1_sd.append(depth1)
        y_depth2_sd.append(depth2)
        z_sd.append(2.5)
        
        weight_sd.append(0.1) # 1/10
'''
# d) midgame+endgame at depth vs depth => stdev = 0
for n_discs in range(6, 65):
    for depth in (0, 64-n_discs+1):
    
        w_n_discs_sd.append(n_discs)
        x_depth1_sd.append(depth)
        y_depth2_sd.append(depth)

        z_sd.append(0)

        #poids (forcement tres precis)
        weight_sd.append(2) # 2/1
'''        
'''

for n_discs in range(29, 40):
    depth2 = 64-n_discs
    depth1 = n_discs & 1
    
    if len(data[n_discs][depth1][depth2]) > 2:
        print('n_discs', n_discs, 'depth1', depth1, 'depth2', depth2, 'sd', statistics.stdev(data[n_discs][depth1][depth2]), 'n_data', len(data[n_discs][depth1][depth2]))
'''

# la fonction

# 3 D
def f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g):
    w, x, y = wxy
    w = 64-w
    res = probcut_a * w + probcut_b * x + probcut_c * y
    res = probcut_d * res**3 + probcut_e * res**2 + probcut_f * res + probcut_g
    return res

# bornage de la fonction
def f_max(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g):
    return np.minimum(40.0, np.maximum(0.0, f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g)))

# calcul des parametres de la fonction
# 3 D
popt_sd, pcov_sd = curve_fit(f, (w_n_discs_sd, x_depth1_sd, y_depth2_sd), z_sd, np.ones(7), sigma=weight_sd, absolute_sigma=True,  maxfev=10000)


'''
# 2D
def f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f):
    w, x, y = wxy
    w = 64-w
    res = probcut_a * w + probcut_b * x + probcut_c * y
    res = probcut_d * res * res + probcut_e * res + probcut_f
    return res

def f_max(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f):
        return np.minimum(40.0, np.maximum(-2.0, f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f)))

# calcul des parametres de la fonction        
# 2 D
popt_sd, pcov_sd = curve_fit(f, (w_n_discs_sd, x_depth1_sd, y_depth2_sd), z_sd, np.ones(6), sigma=weight_sd, absolute_sigma=True,  maxfev=6000)

'''


# affichage des parametres
for i in range(len(popt_sd)):
    print('constexpr double probcut_' + chr(ord('a') + i) + ' = ' + str(popt_sd[i]) + ';')

for i in range(len(popt_sd)):
    print(str(popt_sd[i]).replace('.', ','))
    
    
# affichage graphique du resultat
def plot_fit_result_onephase(w, x, y, z, n_discs, params):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    phase = 64-n_discs
    x_depth1_phase = []
    y_depth2_phase = []
    z_error_phase = []
    for ww, xx, yy, zz in zip(w, x, y, z):
        if ww == n_discs:
            x_depth1_phase.append(xx)
            y_depth2_phase.append(yy)
            z_error_phase.append(zz)
    ax.plot(x_depth1_phase, y_depth2_phase, z_error_phase, ms=3, marker="o",linestyle='None', label=f"nb empty {phase}")
    mx, my = np.meshgrid(range(phase), range(phase))
    ax.plot_wireframe(mx, my, f_max((n_discs, mx, my), *params), rstride=4, cstride=2)
    ax.set_xlabel('shallow depth')
    ax.set_ylabel('depth')
    ax.set_zlabel('error')

    ax.set_xlim((0, phase))
    ax.set_ylim((0, phase))
    ax.set_zlim((0, phase))
    ax.legend()
    plt.show()


# representation graphique de la fonction a 14, 24, 34, 44 discs
for i in [14, 24, 34, 44]:
    plot_fit_result_onephase(w_n_discs_sd, x_depth1_sd, y_depth2_sd, z_sd, i, popt_sd)


'''
def plot_fit_result_allphases(w, x, y, z, params):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']  # liste des couleurs par défaut

    for i, n_moves in enumerate(range(10, 60, 10)):
        n_discs = 4 + n_moves
        x_depth1_phase = []
        y_depth2_phase = []
        z_error_phase = []
        for ww, xx, yy, zz in zip(w, x, y, z):
            if ww == n_discs:
                x_depth1_phase.append(xx)
                y_depth2_phase.append(yy)
                z_error_phase.append(zz)

        # Récupère la couleur suivante dans le cycle
        color = color_cycler[i % len(color_cycler)]

        ax.plot(
            x_depth1_phase, y_depth2_phase, z_error_phase,
            ms=5, marker="o", alpha=1.0, linestyle='None',
            label=f'n_moves={n_moves}', color=color
        )

        n_remaining_moves = 60 - n_moves
        mx, my = np.meshgrid(range(30), range(n_remaining_moves))
        ax.plot_wireframe(mx, my, f_max((n_discs, mx, my), *params),
                          rstride=4, cstride=2, alpha=0.5, color=color)

    ax.set_xlabel('depth1_short')
    ax.set_ylabel('depth2_long')
    ax.set_zlabel('error')
    ax.set_xlim((0, 30))
    ax.set_ylim((0, 60))
    ax.set_zlim((0, 40))
    ax.legend()
    plt.show()
    
plot_fit_result_allphases(w_n_discs_sd, x_depth1_sd, y_depth2_sd, z_sd, popt_sd)
'''
