from matplotlib import pyplot as plt
import numpy as np
class NetworkVisualizer:
    def __init__(self, neural_network):
        """
        neural_network: instance de NeuralNetwork à visualiser
        """
        self.nn = neural_network

    def setup_live_plot(self):
        plt.ion()
        rows, cols = 8, 16
        self.fig, self.axes = plt.subplots(rows, cols, figsize=(16, 8))
        self.fig.suptitle('Apprentissage en temps réel - Évolution des neurones', fontsize=16)
        
        self.images = []
        for i in range(128):
            row = i // cols
            col = i % cols
            ax = self.axes[row, col]
            
            neuron_image = self.nn.w1[i].reshape(28, 28)
            vmax = np.abs(self.nn.w1).max()
            
            im = ax.imshow(neuron_image, cmap='seismic', vmin=-vmax, vmax=vmax)
            self.images.append(im)
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.show(block=False)


    def update_w1_live(self, epoch=None, loss=None, accuracy=None):
        """Met à jour la visualisation avec les poids actuels"""
        vmax = np.abs(self.nn.w1).max()
        
        for i in range(128):
            neuron_image = self.nn.w1[i].reshape(28, 28)
            self.images[i].set_data(neuron_image)
            self.images[i].set_clim(vmin=-vmax, vmax=vmax)
        
        if epoch is not None:
            title = f'Epoch {epoch}'
            if loss is not None:
                title += f' - Loss: {loss:.4f}'
            if accuracy is not None:
                title += f' - Acc: {accuracy:.2f}%'
            self.fig.suptitle(title, fontsize=16)
        
        plt.pause(0.01)


    def close_live_plot(self):
        """Ferme proprement la visualisation"""
        plt.ioff()
        plt.show()

    def visualize_w1(self):
        """Affiche les 128 neurones de la couche cachée"""

        rows, cols = 8, 16
        fig, axes = plt.subplots(rows, cols, figsize=(16, 8))

        for i in range (self.nn.w1.shape[0]):

            row = i // cols
            col = i % cols
            ax = axes[row, col]

            neuron = self.nn.w1[i]
            neuron_image = neuron.reshape(28, 28)

            vmax = np.abs(self.nn.w1).max()
            ax.imshow(neuron_image, cmap='seismic', vmin=-vmax, vmax=vmax)
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        
    def visualize_w2_composite(self):
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Vision composite du réseau pour chaque chiffre', fontsize=16)
        
        axes = axes.flatten()
        
        # Reshaper W1 en (128, 28, 28) - chaque neurone est une image
        w1_images = self.nn.w1.reshape(128, 28, 28)
        
        # Pour chaque chiffre
        for digit in range(10):
            ax = axes[digit]
            
            # Récupérer les poids pour ce chiffre (128 valeurs)
            weights = self.nn.w2[digit, :]  # Shape: (128,)
            
            # Créer l'image composite : somme pondérée
            # weights: (128,) × w1_images: (128, 28, 28) → (28, 28)
            composite_image = np.tensordot(weights, w1_images, axes=([0], [0]))
            
            # Afficher
            vmax = np.abs(composite_image).max()
            ax.imshow(composite_image, cmap='seismic', vmin=-vmax, vmax=vmax)
            ax.set_title(f'{digit}', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_loss_curve(self, losses):
        """Affiche la courbe de loss"""
        pass  # Pour plus tard

