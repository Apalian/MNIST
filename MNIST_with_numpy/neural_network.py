import numpy as np

class NeuralNetwork :

    def __init__(self, input_size, hidden_size, output_size, learning_rate):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        #Intialisation des paramètres des couches
        #Layer 1 paramètre w est un matruice de taille (128x784) car on veut passer d'un vecteur de 784pixels a 128 valeurs 
        self.w1 = np.random.randn(self.hidden_size, self.input_size)
        # Réduction de la valeur des poids
        self.w1 *= 0.01
        #Layer 2 parametre b est une matrice de taille (128x1) car ce sont les biais affectés a chaque neuronnes (128) initialisé a 0, début
        self.b1 = np.zeros((self.hidden_size, 1))

        #Layer 2 paramètre w est un matruice de taille (10x128) car on veut passer d'un vecteur de 128 valeurs a une sortie de 10 valeurs possible (0 à 10) 
        self.w2 = np.random.randn(self.output_size, self.hidden_size)
        # Idem Layer1
        self.w2 *= 0.01
        # Idem Layer1
        self.b2 = np.zeros((output_size, 1))

    def forward(self, X):
        # Sortie de la premiere couche
        self.z1 = self.w1 @ X + self.b1
        # Fonction d'activation reLU
        self.a1 = np.maximum(0, self.z1)
        #Sotie de la seconde couche
        self.z2 = self.w2 @ self.a1 + self.b2
        # Fonction d'activaiton softmax
        z2_temp = self.z2 - np.max(self.z2, axis=0, keepdims=True)
        z2_exp = np.exp(z2_temp)
        self.a2 = z2_exp / np.sum(z2_exp, axis=0, keepdims=True)
        return self.a2

    def backward(self, X, Y_true):
        batch_size = X.shape[1]

        # Gradient de l'erreur
        dz2 = self.a2 - Y_true
        
        # Gradient de w2
        self.dw2 = ( dz2 @ self.a1.T ) / batch_size
        # Gradient de b2
        self.db2 = np.sum(dz2, axis=1, keepdims=True) / batch_size

        # Propagaiton du gradient sur a1
        da1 = self.w2.T @ dz2
        # Gradient de z1
        dz1 = da1 * (self.z1 > 0)

        # Gradient de w1
        self.dw1 = (dz1 @ X.T) / batch_size
        # Gradient de db2
        self.db1 = np.sum(dz1, axis=1, keepdims=True) / batch_size

    def update(self):
        self.w1 -= (self.learning_rate * self.dw1)
        self.b1 -= (self.learning_rate * self.db1)
        self.w2 -= (self.learning_rate * self.dw2)
        self.b2 -= (self.learning_rate * self.db2)

    def train(self, X_train, Y_train, X_test, Y_test, epochs=10, batch_size=32, visualizer=None):
        n_samples = X_train.shape[1]
        
        if visualizer:
            visualizer.setup_live_plot()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, n_samples, batch_size):
                x_batch = X_train[:, i:i+batch_size]
                y_batch = Y_train[:, i:i+batch_size]
                
                y_pred = self.forward(x_batch)
                self.backward(x_batch, y_batch)
                self.update()
                
                batch_loss = -np.sum(y_batch * np.log(y_pred + 1e-8))
                total_loss += batch_loss
            
            correct = 0
            n_test = X_test.shape[1]
            
            for i in range(n_test):
                x_test = X_test[:, i:i+1]
                y_test = Y_test[:, i:i+1]
                
                prediction = self.forward(x_test)
                
                if np.argmax(prediction) == np.argmax(y_test):
                    correct += 1
            
            accuracy = correct / n_test * 100
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/n_samples:.4f} - Test Accuracy: {accuracy:.2f}%")
            
            if visualizer:
                visualizer.update_w1_live(epoch+1, total_loss/n_samples, accuracy)
        
        if visualizer:
            visualizer.close_live_plot()