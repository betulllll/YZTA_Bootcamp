import os
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Modeli yükle
T_Model = YOLO('yolov8n.pt')

# Dataset yolu
yaml_file_path = r"C:\Users\mehmet\Desktop\archive\BrainTumor\BrainTumorYolov8\data.yaml"

if __name__ == "__main__":
    # Eğitim işlemi
    results = T_Model.train(
        data=yaml_file_path,
        epochs=50,
        patience=20,
        batch=-1,
        optimizer='auto'
    )
