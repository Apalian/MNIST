import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

from model import Model


class ImageProcessor:
    @staticmethod
    def canvas_to_array(canvas_image):
        """Convertit l'image PIL du canvas en array numpy"""
        img = canvas_image.resize((28, 28), Image.Resampling.LANCZOS)
        img = img.convert('L')
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 784)
        return img_array


class DrawingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.model = Model()
        self.last_x = None
        self.last_y = None
        
        # Créer une image PIL pour dessiner
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Digit Recognizer")
        self.root.geometry("400x600")
        
        # Titre
        title = tk.Label(self.root, text="Draw a digit (0-9)", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, width=280, height=280, 
                                bg='white', cursor='pencil')
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_draw)
        
        # Frame pour les boutons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.clear_button = tk.Button(button_frame, text="Clear", 
                                    command=self.clear_canvas, width=10)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.predict_button = tk.Button(button_frame, text="Predict", 
                                        command=self.predict_digit, width=10)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # Résultat
        self.result_label = tk.Label(self.root, text="", 
                                    font=("Arial", 14), 
                                    justify=tk.LEFT)
        self.result_label.pack(pady=10)

    def paint(self, event):
        x, y = event.x, event.y
        
        # Dessiner sur le canvas
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                width=15, fill='black', 
                                capstyle=tk.ROUND, smooth=True)
            # Dessiner aussi sur l'image PIL
            self.draw.line([self.last_x, self.last_y, x, y], 
                          fill='black', width=15)
        
        self.last_x = x
        self.last_y = y

    def reset_draw(self, event):
        self.last_x = None
        self.last_y = None

    def predict_digit(self):
        # Inverser l'image (noir sur blanc -> blanc sur noir)
        inverted = Image.eval(self.image, lambda x: 255 - x)
        
        # Convertir en array
        img_array = ImageProcessor.canvas_to_array(inverted)
        
        # Prédiction
        prediction = self.model.predict(img_array)
        probas = self.model.predict_proba(img_array)[0]
        
        # Afficher le résultat
        result_text = f"Predicted Digit: {prediction[0]}\n\n"
        
        # Top 3 probabilités
        top_3 = np.argsort(probas)[-3:][::-1]
        result_text += "Probabilities:\n"
        for digit in top_3:
            result_text += f"  {digit}: {probas[digit]*100:.1f}%\n"
        
        self.result_label.config(text=result_text)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="")
        self.last_x = None
        self.last_y = None
        
        # Réinitialiser l'image PIL
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = DrawingApp()
    print("Training model...")
    app.model.train(epochs=5)
    print("Model trained! You can now draw.")
    app.run()