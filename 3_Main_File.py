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
train_df, valid_df = train_test_split(
    train_df, test_size=0.20, stratify=train_df["label"]
)

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


# & ____________________________ Train ve Test Setlerindeki Kanser Dağılımları _________________________________
# * 7-) --------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt


def create_pie_chart(ax, data, title, colors, explode):
    ax.pie(
        data.values(),
        labels=data.keys(),
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=explode,
    )
    ax.axis("equal")
    ax.set_title(title, weight="bold")


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot for train_data
label_counts_train = train_df["label"].value_counts().to_dict()
total_count_train = sum(label_counts_train.values())
label_percentages_train = {
    label: count / total_count_train * 100
    for label, count in label_counts_train.items()
}
# Updated colors and explode for "axial", "saggital", "notumor" classes
colors_train = ["seagreen", "slategrey", "lightcoral"]
explode_train = [
    0.1 if label == "notumor" else 0 for label in label_counts_train.keys()
]
create_pie_chart(
    axes[0],
    label_percentages_train,
    "Train Data Distribution",
    colors_train,
    explode_train,
)

# Plot for test_data
label_counts_test = test_df["label"].value_counts().to_dict()
total_count_test = sum(label_counts_test.values())
label_percentages_test = {
    label: count / total_count_test * 100 for label, count in label_counts_test.items()
}
# Updated colors and explode for "axial", "saggital", "notumor" classes
colors_test = ["seagreen", "slategrey", "lightcoral"]
explode_test = [0.1 if label == "notumor" else 0 for label in label_counts_test.keys()]
create_pie_chart(
    axes[1], label_percentages_test, "Test Data Distribution", colors_test, explode_test
)

plt.tight_layout()
plt.savefig("TrainTestDataDistribution.png")
print("Saved -- > Train_Test_Data_Distribution.png")
plt.close()


# & ______________________________________ DATA AUGMENTATION ___________________________________________________
# * 8-) --------------------------------------------------------------------------------------------------------

image_size = [150, 150]  # Her görüntü boyutunun standart olmasını sağladık.

batch_size = 64
color_channel = 3
image_shape = (150, 150, 3)

train_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_generator = ImageDataGenerator(rescale=1.0 / 255)

train_data_generator = train_generator.flow_from_dataframe(
    train_df,
    x_col="image_data",
    y_col="label",
    target_size=image_size,
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size,
)

test_data_generator = test_generator.flow_from_dataframe(
    test_df,
    x_col="image_data",
    y_col="label",
    target_size=image_size,
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size,
)

valid_data_generator = test_generator.flow_from_dataframe(
    valid_df,
    x_col="image_data",
    y_col="label",
    target_size=image_size,
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size,
)


""" def load_images_from_folder(folder, target_size=(150, 150)):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)
                img = img / 255.0  # Normalizasyon
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)


train_images, train_labels = load_images_from_folder(train_data)
test_images, test_labels = load_images_from_folder(test_data)
valid_images, valid_labels = load_images_from_folder(valid_data)

from sklearn.preprocessing import LabelBinarizer


lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)
valid_labels = lb.transform(valid_labels)

num_classes = len(lb.classes_)  # Doğru sınıf sayısını alıyoruz
 """

# & _________________________________________ Model Development ________________________________________________
# * 9-) --------------------------------------------------------------------------------------------------------

image_shape = (128, 128, 3)
# image_shape = (150, 150, 3)
batch_size = 64


def create_densenet169_model(input_shape, num_classes):
    base_model = DenseNet169(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model


num_classes = len(train_data_generator.class_indices)


model = create_densenet169_model(image_shape, num_classes)


# def create_densenet169_model(input_shape, num_classes):
#     base_model = DenseNet169(
#         weights="imagenet", include_top=False, input_shape=input_shape
#     )

#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation="relu")(x)
#     predictions = Dense(num_classes, activation="softmax")(x)

#     model = Model(inputs=base_model.input, outputs=predictions)

#     for layer in base_model.layers:
#         layer.trainable = False

#     return model


# model = create_densenet169_model(image_shape, num_classes)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


print(model.summary())

import sys
from contextlib import redirect_stdout

# Define the path where you want to save the summary
summary_file_path = "model_summary.txt"

# Save the model summary to a text file
with open(summary_file_path, "w") as f:
    with redirect_stdout(f):
        model.summary()

print(f"Model summary saved to --> {summary_file_path}")


# & _________________________________ MODELİ EĞİTME & CALLBACK _________________________________________
# * 10-) ------------------------------------------------------------------------------------------------
# EarlyStopping ve ReduceLROnPlateau, eğitim sürecinde modelin performansını izlemek
# ve optimize etmek için kullanılan callback fonksiyonlarıdır.


# ? Eger en iyi modeli 'BestModels/bestModel-01-0.86.model' şeklinde kaydetmek istersek
# ? bu bloğu yorum satirindan cikarmali ve histor degiskeninde yer alan callbacs'in karsiligi olarak kullanmaliyiz.

# En iyi modelin kaydedilmesi için ModelCheckpoint callback'ini oluşturun

ckp_interval = 5 * int(np.ceil(train_df.shape[0] / batch_size))  # x * epocs
# ckp_folder = r"C:/Users/ziyak/OneDrive/Documents/GitHub/NeuroDenseNet_StableProject/ModelCheckpoints"
ckp_folder = r"C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/ModelCheckPoints_MS"
ckp_path = os.path.join(
    ckp_folder, r"epocch_{epoch:02d}_val_acc_{val_accuracy:.2f}.hdf5"
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=ckp_path,
    save_weights_only=True,
    monitor="val_loss",  # Bu satırı güncelledik
    verbose=1,
    save_best_only=True,
)


callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6),
    checkpoint,  # Bu satırı güncelledik
]

history = model.fit(
    train_data_generator,
    batch_size=batch_size,
    validation_data=valid_data_generator,
    epochs=20,
    callbacks=callbacks,
)

""" history = model.fit(
    train_images,
    train_labels,
    validation_data=(valid_images, valid_labels),
    epochs=50,
    batch_size=batch_size,
    callbacks=callbacks,
) """


# & ___________________________ History Dosyasının Kayıt Edilmesi ______________________________________
# * 11-) ------------------------------------------------------------------------------------------------

print("__________HISTORY KAYDEDILIYOR_____________")

# History verisini bir dosyaya kaydedin
with open("history_MS.npy", "wb") as f:
    np.save(f, history.history)

# & _________________________ Eğitim Bitiminde Modelin Kayıt Edilmesi __________________________________
# * 12-) ------------------------------------------------------------------------------------------------

model.save("MS_classifier_MS.keras")
