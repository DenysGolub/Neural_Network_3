import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from data_helper import DataHelper
from tkinter import messagebox
from neural_network import NeuralNetwork
import numpy as np

import numpy as np
import data_helper
import cv2
from layer import FullyConnectedLayer, FlattenLayer, ActivationLayer, Softmax
from activations import Activation
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import os
class SimpleInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Завантаження моделі та зображення")
        
        # Кнопка для завантаження моделі
        self.load_model_button = tk.Button(root, text="Завантажити модель", command=self.load_model)
        self.load_model_button.pack(pady=10)
        
        # Кнопка для завантаження фото
        self.load_image_button = tk.Button(root, text="Завантажити фото", command=self.load_image)
        self.load_image_button.pack(pady=10)
        
        # Полотно для відображення зображення
        self.canvas = tk.Canvas(root, width=300, height=300, bg="gray")
        self.canvas.pack(pady=10)
        
        # Змінна для зберігання об'єкта зображення
        self.image = None

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Виберіть файл моделі")
        if model_path:
            print(f"Модель завантажена з: {model_path}")
            self.loaded_network = None
            import pandas as pd

            self.loaded_network = data_helper.DataHelper().import_network(model_path)
            print()

    def load_image(self):
        image_path = filedialog.askopenfilename(title="Виберіть зображення", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if image_path:
            image = Image.open(image_path)
            image = image.resize((300, 300))  # Зміна розміру зображення для Canvas
            self.image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)
            print(f"Зображення завантажено з: {image_path}")
            pred = self.loaded_network.predict(DataHelper.load_resized_gray_image(IMG_SIZE=16, path=image_path))[0]
            idx = np.argmax(pred)
            messagebox.showinfo("Result", self.loaded_network.classes[idx])

# Ініціалізація інтерфейсу
root = tk.Tk()
app = SimpleInterface(root)
root.mainloop()
