import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

class StdDevVisualizer:
    def __init__(self, data):
        self.data = data
        self.window = tk.Tk()
        self.window.title("Visualisateur de Std Dev")
        self.window.geometry("1200x700")
        
        # Frame pour les contrôles
        control_frame = ttk.Frame(self.window, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Sélection n_discs
        ttk.Label(control_frame, text="n_discs:").grid(row=0, column=0, padx=5, pady=5)
        self.n_discs_var = tk.StringVar(value="8")
        n_discs_spinbox = ttk.Spinbox(
            control_frame, 
            from_=8, 
            to=40, 
            textvariable=self.n_discs_var,
            width=10,
            command=self.update_shallow_depth_range
        )
        n_discs_spinbox.grid(row=0, column=1, padx=5, pady=5)
        
        # Sélection shallow_depth
        ttk.Label(control_frame, text="shallow_depth:").grid(row=0, column=2, padx=5, pady=5)
        self.shallow_depth_var = tk.StringVar(value="0")
        self.shallow_depth_spinbox = ttk.Spinbox(
            control_frame,
            from_=0,
            to=20,
            textvariable=self.shallow_depth_var,
            width=10
        )
        self.shallow_depth_spinbox.grid(row=0, column=3, padx=5, pady=5)
        
        # Bouton de mise à jour
        ttk.Button(control_frame, text="Afficher", command=self.plot_data).grid(
            row=0, column=4, padx=10, pady=5
        )
        
        # Bouton de reset zoom
        ttk.Button(control_frame, text="Reset Zoom", command=self.reset_zoom).grid(
            row=0, column=5, padx=10, pady=5
        )
        
        # Label d'information
        self.info_label = ttk.Label(control_frame, text="", foreground="blue")
        self.info_label.grid(row=1, column=0, columnspan=6, pady=5)
        
        # Frame pour le graphique
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration du grid
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        
        # Initialiser la plage de shallow_depth
        self.update_shallow_depth_range()
        
        # Afficher les données initiales
        self.plot_data()
    
    def update_shallow_depth_range(self):
        """Met à jour la plage de shallow_depth en fonction de n_discs"""
        try:
            n_discs = int(self.n_discs_var.get())
            parity = n_discs & 1
            max_depth = 20 + parity
            
            # shallow_depth va de (0 + parity) à (max_depth - 2)
            min_shallow = parity
            max_shallow = max_depth - 2
            
            self.shallow_depth_spinbox.config(from_=min_shallow, to=max_shallow)
            
            # Ajuster la valeur actuelle si nécessaire
            current_shallow = int(self.shallow_depth_var.get())
            if current_shallow < min_shallow:
                self.shallow_depth_var.set(str(min_shallow))
            elif current_shallow > max_shallow:
                self.shallow_depth_var.set(str(max_shallow))
                
        except ValueError:
            pass
    
    def plot_data(self):
        """Trace les courbes de std dev en fonction de depth"""
        try:
            n_discs = int(self.n_discs_var.get())
            shallow_depth = int(self.shallow_depth_var.get())
            
            # Effacer le graphique précédent
            self.ax.clear()
            
            # Collecter les données
            depths = []
            stddevs = []
            counts = []
            
            # Calculer la parité de n_discs
            parity = n_discs & 1
            
            # Commencer à shallow_depth + 2 et respecter la parité
            start_depth = shallow_depth + 2
            # Ajuster start_depth pour respecter la parité
            if (start_depth & 1) != parity:
                start_depth += 1
            
            for depth in range(start_depth, 61, 2):  # Incrément de 2 pour garder la parité
                errors = self.data[n_discs][shallow_depth][depth]
                if len(errors) > 0:
                    depths.append(depth)
                    stddevs.append(np.std(errors))
                    counts.append(len(errors))
            
            if len(depths) == 0:
                self.info_label.config(
                    text=f"Aucune donnée pour n_discs={n_discs}, shallow_depth={shallow_depth}",
                    foreground="red"
                )
                self.ax.text(0.5, 0.5, 'Aucune donnée disponible', 
                           ha='center', va='center', fontsize=16,
                           transform=self.ax.transAxes)
                self.canvas.draw()
                return
            
            depths = np.array(depths)
            stddevs = np.array(stddevs)
            counts = np.array(counts)
            
            # Tracer la courbe principale
            self.ax.plot(depths, stddevs, 'o-', linewidth=2, markersize=6, 
                        label='Std Dev', color='#2E86AB')
            
            # Ajouter une courbe de tendance
            if len(depths) > 2:
                z = np.polyfit(depths, stddevs, min(3, len(depths)-1))
                p = np.poly1d(z)
                depths_smooth = np.linspace(depths.min(), depths.max(), 100)
                self.ax.plot(depths_smooth, p(depths_smooth), '--', 
                           alpha=0.5, color='#A23B72', label='Tendance')
            
            # Ajouter les annotations pour les points
            for i, (d, s, c) in enumerate(zip(depths, stddevs, counts)):
                if i % max(1, len(depths)//10) == 0:  # Annoter 1 point sur 10 max
                    self.ax.annotate(f'{s:.2f}\n(n={c})', 
                               xy=(d, s), 
                               xytext=(5, 5),
                               textcoords='offset points',
                               fontsize=8,
                               alpha=0.7)
            
            # Configuration du graphique
            self.ax.set_xlabel('Depth', fontsize=12, fontweight='bold')
            self.ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
            self.ax.set_title(
                f'Évolution de la Std Dev pour n_discs={n_discs}, shallow_depth={shallow_depth}',
                fontsize=14, fontweight='bold', pad=20
            )
            self.ax.grid(True, alpha=0.3, linestyle='--')
            self.ax.legend(loc='best', fontsize=10)
            
            # Informations statistiques
            delta_depths = depths - shallow_depth
            info_text = (f"Données : {len(depths)} points | "
                        f"Delta depth : [{delta_depths.min()}-{delta_depths.max()}] | "
                        f"Std Dev : [{stddevs.min():.2f}-{stddevs.max():.2f}] | "
                        f"Moyenne : {stddevs.mean():.2f}")
            self.info_label.config(text=info_text, foreground="blue")
            
            # Ajuster les marges
            self.fig.tight_layout()
            self.canvas.draw()
            
        except ValueError as e:
            self.info_label.config(text=f"Erreur: {e}", foreground="red")
    
    def reset_zoom(self):
        """Réinitialise le zoom du graphique"""
        self.ax.autoscale()
        self.canvas.draw()
    
    def run(self):
        """Lance l'interface"""
        self.window.mainloop()


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

print("Chargement des données...")
data_files = ['../data/midgame/probcut_mid.txt', '../data/midgame/probcut_mid1.txt']
data_files_end = ['../data/endgame/probcut_end.txt', '../data/endgame/probcut_end_ext.txt']

data = [[[[] for _ in range(61)] for _ in range(61)] for _ in range(65)]

for data_file in data_files:
    try:
        with open(data_file, 'r') as f:
            print(f'Lecture de {data_file}')
            raw_data = f.read().splitlines()
        
        for datum in raw_data:
            n_discs, depth1, depth2, error = [int(elem) for elem in datum.split()]
            data[n_discs][depth1][depth2].append(error)
            
    except (FileNotFoundError, IOError) as e:
        print(f'Impossible d\'ouvrir {data_file}: {e}')

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


print("Données chargées !")

# Statistiques rapides
total_triplets = 0
for n in range(8, 41):
    for s in range(61):
        for d in range(61):
            if len(data[n][s][d]) > 0:
                total_triplets += 1

print(f"Nombre total de triplets avec données : {total_triplets}")

# ============================================================================
# LANCER LE VISUALISATEUR
# ============================================================================

visualizer = StdDevVisualizer(data)
visualizer.run()
