#
# * 1-) --------------------------------------------------------------------------------------------------------
# Kütüphanelerimizi aşağıdaki gibi import edelim.
# Data Manipulation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd

# Data preprocessing
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# from PIL import Image, ImageEnhance
from sklearn.utils import shuffle

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# For ML Models
from keras.applications import Xception, ResNet50, MobileNet, VGG16
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.losses import *
from keras.models import *
from keras.metrics import *
from keras.optimizers import *
from keras.optimizers import Adam
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Miscellaneous
import cv2

# from tqdm import tqdm
import os
import random
import pickle

# & ______________________________________ Görselleştirme Fonksiyonu ___________________________________________
# * 2-) --------------------------------------------------------------------------------------------------------


def visualize_images(image_path, title, save_name):
    image_files = os.listdir(image_path)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        img_path = os.path.join(image_path, image_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(5, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    print(f"Saved -- > {save_name}")


# gliomaPath = "C:/Users/ziyak/OneDrive/Documents/GitHub/NeuroDenseNet169Project/DataSet/Training/glioma/"
cropped_images_axial = "C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/DataSet/Train/Axial"
visualize_images(cropped_images_axial, "Sample Axial Images", "Sample_Axial_Images.png")

# meningiomaPath = "C:/Users/ziyak/OneDrive/Documents/GitHub/NeuroDenseNet169Project/DataSet/Training/meningioma/"
cropped_images_notumor = "C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/DataSet/Train/notumor"
visualize_images(
    cropped_images_notumor, "Sample notumor Images", "Sample_notumor_Images.png"
)

# notumorPath = "C:/Users/ziyak/OneDrive/Documents/GitHub/NeuroDenseNet169Project/DataSet/Training/notumor/"
cropped_images_sagittal = "C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/DataSet/Train/Sagittal"
visualize_images(
    cropped_images_sagittal, "Sample Sagittal Images", "Sample_Sagittal_Images.png"
)


# & ____________________________________ DataFrame Oluşturma Fonksiyonu ________________________________________
# * 3-) --------------------------------------------------------------------------------------------------------
# Train, test ve validation dataların path'ini aşağıdaki ilgili değişkenlere atayalım.


def create_dataframe(data_path):
    filepath = []
    label = []
    image_folder = os.listdir(data_path)
    for folder in image_folder:
        folder_path = os.path.join(data_path, folder)
        filelist = os.listdir(folder_path)
        for file in filelist:
            new_path = os.path.join(folder_path, file)
            filepath.append(new_path)
            label.append(folder)
    image_data = pd.Series(filepath, name="image_data")
    label_data = pd.Series(label, name="label")
    df = pd.concat([image_data, label_data], axis=1)
    return df


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# & ____________________________________ DataFramelerin Elde Edilmesi __________________________________________
# * 4-) --------------------------------------------------------------------------------------------------------

# train_data = "C:/Users/ziyak/OneDrive/Documents/GitHub/NeuroDenseNet169Project/Cropped_DataSet/Training"
# test_data = "C:/Users/ziyak/OneDrive/Documents/GitHub/NeuroDenseNet169Project/Cropped_DataSet/Testing"
# valid_data = "C:/Users/ziyak/OneDrive/Documents/GitHub/NeuroDenseNet169Project/Cropped_DataSet/Testing"


train_data = "Augmented_Images/Train"
test_data = "Augmented_Images/Test"
valid_data = "Augmented_Images/Test"

train_df = create_dataframe(train_data)
test_df = create_dataframe(test_data)
# valid_df = create_dataframe(valid_data)

# & __________________________ Train Set ve Validation Set'in %75 - %25 Ayrıştırma _____________________________
# * 5-) --------------------------------------------------------------------------------------------------------


# Verileri eğitim ve doğrulama olarak ayırın
valid_df, test_df = train_test_split(test_df, test_size=0.20, stratify=test_df["label"])


print("Eğitim seti boyutları:", train_df.shape)
print("Doğrulama seti boyutları:", valid_df.shape)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("Train DataFrame:\n", train_df.head())
print("Test DataFrame:\n", test_df.head())
print("Validation DataFrame:\n", valid_df.head())

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# & ___________________________ Betimsel İstatistik Fonksiyonu ve Elde Edilmesi _________________________________
# * 6-) --------------------------------------------------------------------------------------------------------


# _____________ Train, Test & Validation Seti İçin Betimsel İstatistik _____________
def print_dataset_statistics(df, name):
    print(f"{name} DataFrame:\n", df.head())
    print(f"{name} seti boyutları--> \n", df.shape)
    print(f"Eksik veri gözlemleri--> \n", df.isnull().sum())
    print(f"Kanser Türü Sayıları--> \n", df["label"].value_counts())


print_dataset_statistics(train_df, "Eğitim")
print_dataset_statistics(test_df, "Test")
print_dataset_statistics(valid_df, "Validasyon")
