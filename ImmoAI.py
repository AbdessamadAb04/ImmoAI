import tkinter as tk 
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statsmodels.tsa.arima.model import ARIMA as ARIMA
import threading

class RealEstateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Estimation des Prix Immobiliers - Dashboard IA")
        self.root.geometry("1200x800")
        self.root.configure(bg="white")
        self.root.minsize(1000, 700)
        
        # Create home page first
        self.create_home_page()
        
        # Initialize variables for main interface
        self.min_val = tk.IntVar(value=1)
        self.max_val = tk.IntVar(value=100)
        self.n_clusters = tk.IntVar(value=3)
        self.is_running = False
        
        # Generate sample data
        self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample real estate data"""
        np.random.seed(42)
        self.surface = np.random.randint(50, 200, 100)  # Surface en m²
        self.nb_pieces = np.random.randint(1, 6, 100)   # Nombre de pièces
        self.quartier = np.random.randint(1, 6, 100)    # Qualité du quartier (1-5)
        self.prix = 5000 * self.surface + 20000 * self.nb_pieces + 10000 * self.quartier + np.random.normal(0, 50000, 100)
        self.X_clust = np.column_stack((self.surface, self.prix/1000))
        self.evolution_prix = np.cumsum(np.random.randn(100) * 5000 + 2000)

    def create_home_page(self):
        """Create the home page with welcome message and buttons"""
        self.home_frame = tk.Frame(self.root, bg="white")
        self.home_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        tk.Label(self.home_frame, 
                text="Estimation de Prix Immobiliers par IA",
                font=("Segoe UI", 26, "bold"), 
                bg="white", fg="#2c3e50").pack(pady=60)

        # Subtitle
        tk.Label(self.home_frame, 
                text="By Abdessamad Abounouh",
                font=("Segoe UI", 14, "italic"), 
                bg="white", fg="#5D6D7E").pack(pady=10)

        # Dashboard button
        tk.Button(self.home_frame, 
                 text="Accéder au Dashboard",
                 command=self.show_main_interface,
                 bg="#27ae60", fg="white",
                 font=('Segoe UI', 14, 'bold'), 
                 padx=25, pady=12,
                 relief=tk.FLAT, cursor="hand2").pack(pady=30)

        # Exit button
        tk.Button(self.home_frame, 
                 text="Quitter",
                 command=self.root.destroy,
                 bg="#ecf0f1", fg="#2c3e50",
                 font=('Segoe UI', 12), 
                 padx=20, pady=10,
                 relief=tk.FLAT, cursor="hand2").pack()

    def show_main_interface(self):
        """Switch from home page to main interface"""
        self.home_frame.pack_forget()
        self.setup_main_interface()

    def setup_main_interface(self):
        """Create the main application interface"""
        # Main container
        self.main_frame = tk.Frame(self.root, bg="white")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(self.main_frame, bg="white")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(header_frame, 
                text="EMSI - Estimation Immobilière par IA", 
                font=("Segoe UI", 18, "bold"), 
                bg="white", fg="#2c3e50").pack(side=tk.LEFT)
        
        # Version badge
        tk.Label(header_frame, 
                text="v1.0.0", 
                font=("Segoe UI", 10, "italic"),
                bg="white", fg="#7f8c8d").pack(side=tk.RIGHT, padx=5)
        
        # Content area
        content_frame = tk.Frame(self.main_frame, bg="white")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Models
        self.models_frame = tk.Frame(content_frame, width=300, bg="white", bd=1, relief=tk.RIDGE)
        self.models_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        self.models_frame.pack_propagate(False)
        
        tk.Label(self.models_frame, 
                text="Paramètres des Modèles", 
                font=("Segoe UI", 12, "bold"),
                bg="white", fg="#2c3e50").pack(pady=(10, 15), anchor='w')
        
        # Parameter cards
        self.create_param_card("Min index (dataset)", 
                             "Index minimal dans le dataset",
                             self.min_val, 1, 99, self.on_min_change)
        
        self.create_param_card("Max index (dataset)", 
                             "Index maximal dans le dataset",
                             self.max_val, 2, 100, self.on_max_change)
        
        self.create_param_card("Nombre de segments (K-Means)", 
                             "Nombre de segments de marché",
                             self.n_clusters, 1, 10)
        
        # Right panel - Output
        self.output_frame = tk.Frame(content_frame, bg="white")
        self.output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Output notebook
        self.output_notebook = ttk.Notebook(self.output_frame)
        self.output_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Console tab
        self.console_tab = tk.Frame(self.output_notebook, bg="white")
        self.output_notebook.add(self.console_tab, text="Console")
        
        self.console_text = scrolledtext.ScrolledText(
            self.console_tab, 
            wrap=tk.WORD, 
            font=('Consolas', 10),
            bg='white', 
            fg='#333333',
            padx=10, 
            pady=10
        )
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        # Graph tab
        self.graph_tab = tk.Frame(self.output_notebook, bg="white")
        self.output_notebook.add(self.graph_tab, text="Visualisation")
        
        self.graph_canvas_frame = tk.Frame(self.graph_tab, bg="white")
        self.graph_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model buttons frame
        self.btn_frame = tk.Frame(self.main_frame, bg="white")
        self.btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Create model buttons
        self.create_model_button("Régression Linéaire", self.threaded(self.run_linear_regression))
        self.create_model_button("Forêt Aléatoire", self.threaded(self.run_random_forest))
        self.create_model_button("Segmentation (K-Means)", self.threaded(self.run_kmeans))
        self.create_model_button("Tendance (ARIMA)", self.threaded(self.run_arima))
        self.create_model_button("Validation Croisée", self.threaded(self.run_cross_validation))
        
        # Footer
        footer_frame = tk.Frame(self.main_frame, bg="white")
        footer_frame.pack(fill=tk.X, pady=(15, 0))
        
        tk.Label(footer_frame, 
                text="© 2025 EMSI - Estimation Immobilière | Développé avec Python",
                font=("Segoe UI", 9),
                bg="white", fg="#7f8c8d").pack(side=tk.RIGHT)
        
        # Save button
        self.save_btn = tk.Button(footer_frame,
                                text="Enregistrer Graphique",
                                command=self.save_figure,
                                bg="#27ae60", fg="white",
                                font=('Segoe UI', 10),
                                padx=15, pady=5,
                                relief=tk.FLAT, cursor="hand2")
        self.save_btn.pack(side=tk.LEFT)
        self.save_btn.config(state='disabled')

    def create_param_card(self, title, description, variable, from_, to, command=None):
        """Create a parameter card with slider"""
        card = tk.Frame(self.models_frame, bg="white", bd=1, relief=tk.RIDGE)
        card.pack(fill=tk.X, pady=5)
        
        # Card content
        content_frame = tk.Frame(card, bg="white")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        tk.Label(content_frame, 
                text=title, 
                font=("Segoe UI", 10, "bold"),
                bg="white", fg="#2c3e50").pack(anchor='w')
        
        # Description
        tk.Label(content_frame, 
                text=description, 
                font=("Segoe UI", 9),
                bg="white", fg="#7f8c8d").pack(anchor='w')
        
        # Slider
        slider_frame = tk.Frame(content_frame, bg="white")
        slider_frame.pack(fill=tk.X, pady=(5, 0))
        
        scale = tk.Scale(slider_frame, 
                        from_=from_, 
                        to=to, 
                        orient=tk.HORIZONTAL,
                        variable=variable,
                        command=command if command else None,
                        bg="white",
                        highlightthickness=0)
        scale.pack(fill=tk.X)
        
        # Current value
        tk.Label(slider_frame, 
                textvariable=variable, 
                font=("Segoe UI", 9),
                bg="white", fg="#7f8c8d").pack(side=tk.RIGHT, padx=5)

    def create_model_button(self, text, command):
        """Create a styled model button"""
        btn = tk.Button(self.btn_frame,
                      text=text,
                      command=command,
                      bg="#27ae60", fg="white",
                      font=('Segoe UI', 10, 'bold'),
                      padx=15, pady=8,
                      relief=tk.FLAT, cursor="hand2")
        btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

    # --- Helper methods ---
    def on_min_change(self, val):
        val = int(float(val))
        if val >= self.max_val.get():
            self.min_val.set(self.max_val.get() - 1)

    def on_max_change(self, val):
        val = int(float(val))
        if val <= self.min_val.get():
            self.max_val.set(self.min_val.get() + 1)

    def clear_output(self):
        """Clear console and graph outputs"""
        self.console_text.delete(1.0, tk.END)
        for widget in self.graph_canvas_frame.winfo_children():
            widget.destroy()
        self.save_btn.config(state='disabled')

    def log_message(self, msg):
        """Add message to console with timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.console_text.see(tk.END)

    def get_min_max_indices(self):
        min_idx = self.min_val.get() - 1
        max_idx = self.max_val.get()
        if min_idx >= max_idx:
            self.log_message("Erreur : min doit être inférieur à max.")
            return 0, len(self.surface)
        if max_idx > len(self.surface):
            max_idx = len(self.surface)
        return min_idx, max_idx

    def show_graph(self, fig):
        """Display a matplotlib figure in the graph tab"""
        for widget in self.graph_canvas_frame.winfo_children():
            widget.destroy()
        
        canvas = FigureCanvasTkAgg(fig, master=self.graph_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.output_notebook.select(self.graph_tab)
        self.save_btn.config(state='normal')

    def disable_buttons(self):
        for btn in self.btn_frame.winfo_children():
            btn.config(state='disabled')
        self.is_running = True

    def enable_buttons(self):
        for btn in self.btn_frame.winfo_children():
            btn.config(state='normal')
        self.is_running = False

    def threaded(self, func):
        """Decorator to run in thread and manage UI"""
        def wrapper():
            if self.is_running:
                messagebox.showwarning("Patientez", "Un calcul est déjà en cours")
                return
            self.disable_buttons()
            threading.Thread(target=self._run_and_enable, args=(func,)).start()
        return wrapper

    def _run_and_enable(self, func):
        try:
            func()
        finally:
            self.enable_buttons()

    # --- Model methods ---
    def run_linear_regression(self):
        self.clear_output()
        self.log_message("Exécution Régression Linéaire (Prix vs Surface)")
        min_idx, max_idx = self.get_min_max_indices()
        self.log_message(f"Paramètres actifs : min={min_idx+1}, max={max_idx}")

        X_sub = self.surface[min_idx:max_idx].reshape(-1, 1)
        y_sub = self.prix[min_idx:max_idx]

        model = LinearRegression()
        model.fit(X_sub, y_sub)
        score = model.score(X_sub, y_sub)
        self.log_message(f"Score R² : {score:.3f}")

        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(X_sub, y_sub/1000, label="Données immobilières", color='#3498db', alpha=0.7, s=50)
        ax.plot(X_sub, model.predict(X_sub)/1000, color='#e74c3c', lw=2.5, label="Régression Linéaire")
        ax.set_title("Estimation Prix vs Surface (Régression Linéaire)", fontsize=14, pad=20)
        ax.set_xlabel('Surface (m²)', fontsize=12)
        ax.set_ylabel('Prix (k€)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add regression equation
        eq_text = f'Prix = {model.coef_[0]/1000:.2f}k€ × Surface + {model.intercept_/1000:.2f}k€'
        ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self.show_graph(fig)

    def run_random_forest(self):
        self.clear_output()
        self.log_message("Exécution Forêt Aléatoire (Prix vs Surface)")
        min_idx, max_idx = self.get_min_max_indices()
        self.log_message(f"Paramètres actifs : min={min_idx+1}, max={max_idx}")
    
        X_sub = self.surface[min_idx:max_idx].reshape(-1, 1)
        y_sub = self.prix[min_idx:max_idx]
    
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_sub, y_sub)
        y_pred = model.predict(X_sub)
        score = model.score(X_sub, y_sub)
        
        # Take first 8 predictions for display
        sample_predictions = y_pred[:8]
        sample_actual = y_sub[:8]
        
        self.log_message(f"Score R² : {score:.3f}")
        self.log_message("\nPrédiction des Prix (k€) - 8 premiers exemples:")
        for i, (actual, pred) in enumerate(zip(sample_actual, sample_predictions)):
            self.log_message(f"Exemple {i+1}: Réel={actual/1000:.1f}k€ | Prédit={pred/1000:.1f}k€")
    
        fig, ax = plt.subplots(figsize=(8,5))
        
        # Create bar positions and width
        bar_width = 0.35
        index = np.arange(len(sample_predictions))
        
        # Plot actual and predicted values as bars
        bars1 = ax.bar(index, sample_actual/1000, bar_width, 
                      color='#3498db', alpha=0.7, label='Prix Réel')
        bars2 = ax.bar(index + bar_width, sample_predictions/1000, bar_width,
                      color='#2ecc71', alpha=0.7, label='Prix Prédit')
        
        ax.set_xlabel("Exemples de biens immobiliers", fontsize=12)
        ax.set_ylabel("Prix (k€)", fontsize=12)
        ax.set_title("Forêt Aléatoire - Comparaison Prix Réels et Prédits", fontsize=14, pad=20)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels([f"Ex {i+1}" for i in range(len(sample_predictions))])
        ax.legend(fontsize=10)
        
        # Add value labels on top of bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom',
                        fontsize=10)
    
        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        self.show_graph(fig)

    def run_kmeans(self):
        self.clear_output()
        self.log_message("Exécution K-Means pour segmentation du marché")
        k = self.n_clusters.get()
        self.log_message(f"Paramètre actif : segments = {k}")

        model = KMeans(n_clusters=k, random_state=42)
        model.fit(self.X_clust)
        labels = model.labels_
        centers = model.cluster_centers_

        fig, ax = plt.subplots(figsize=(8,6))
        
        # Enhanced scatter plot
        scatter = ax.scatter(self.X_clust[:,0], self.X_clust[:,1], c=labels, 
                           cmap='viridis', alpha=0.8, edgecolors='k', s=80)
        
        # Enhanced centroids
        ax.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=200, 
                  linewidths=2, edgecolors='black', label='Centres des segments')
        
        # Add cluster labels
        for i, center in enumerate(centers):
            ax.text(center[0], center[1], f'Segment {i}', fontsize=12, 
                    ha='center', va='center', color='white', 
                    bbox=dict(facecolor='black', alpha=0.7, pad=2))
        
        ax.set_title(f'Segmentation du Marché Immobilier ({k} segments)', fontsize=14, pad=20)
        ax.set_xlabel('Surface (m²)', fontsize=12)
        ax.set_ylabel('Prix (k€)', fontsize=12)
        ax.legend(fontsize=10)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Segment', fontsize=12)
        
        plt.tight_layout()
        self.show_graph(fig)

        self.log_message("Segments du marché affichés sur graphique.")

    def run_arima(self):
        self.clear_output()
        self.log_message("Exécution ARIMA (p=2, d=1, q=2) pour tendance des prix")
        self.log_message("Traitement en cours...")

        try:
            model = ARIMA(self.evolution_prix, order=(2,1,2))
            results = model.fit()
            aic = results.aic
            self.log_message(f"Modèle ARIMA ajusté. AIC = {aic:.3f}")

            fig, ax = plt.subplots(figsize=(10,5))
            
            # Plot historical data
            ax.plot(self.evolution_prix/1000, color='#3498db', linewidth=2, label='Évolution historique des prix')
            
            # Plot fitted values
            ax.plot(results.fittedvalues/1000, color='#e74c3c', linewidth=2, label='Tendance estimée')
            
            ax.set_title('Modèle ARIMA - Tendance des Prix Immobiliers', fontsize=14, pad=20)
            ax.set_xlabel('Période', fontsize=12)
            ax.set_ylabel('Prix (k€)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            self.show_graph(fig)
        except Exception as e:
            self.log_message(f"Erreur ARIMA : {e}")

    def run_cross_validation(self):
        self.clear_output()
        self.log_message("Validation Croisée (Linear Regression vs Random Forest)")
        min_idx, max_idx = self.get_min_max_indices()
        self.log_message(f"Paramètres actifs : min={min_idx+1}, max={max_idx}")

        X_sub = self.surface[min_idx:max_idx].reshape(-1,1)
        y_sub = self.prix[min_idx:max_idx]

        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=50, random_state=42)

        self.log_message("Calcul scores CV ...")
        scores_lr = cross_val_score(lr, X_sub, y_sub, cv=5)
        scores_rf = cross_val_score(rf, X_sub, y_sub, cv=5)

        mean_lr = scores_lr.mean()
        mean_rf = scores_rf.mean()

        self.log_message(f"Score CV LR: {mean_lr:.3f}")
        self.log_message(f"Score CV RF: {mean_rf:.3f}")

        # Graphique
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(['Régression Linéaire', 'Forêt Aléatoire'], [mean_lr, mean_rf], color=['#3498db', '#2ecc71'])
        ax.set_ylim(0,1)
        ax.set_title("Comparaison des Modèles d'Estimation", fontsize=14, pad=20)
        ax.set_xlabel('Modèles', fontsize=12)
        ax.set_ylabel('Score de Validation', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom',
                    fontsize=12)
        
        plt.tight_layout()
        self.show_graph(fig)

    def save_figure(self):
        if not hasattr(self, 'graph_canvas_frame') or not self.graph_canvas_frame.winfo_children():
            messagebox.showinfo("Info", "Aucun graphique à sauvegarder.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                              filetypes=[("PNG files", "*.png"), 
                                                         ("JPEG files", "*.jpg"), 
                                                         ("All files", ".")])
        if filepath:
            # Get the current figure from the canvas
            canvas = self.graph_canvas_frame.winfo_children()[0]
            fig = canvas.figure
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.log_message(f"Graphique sauvegardé dans {filepath}")


if __name__ == "__main__":
    root = tk.Tk()
    app = RealEstateApp(root)
    root.mainloop()