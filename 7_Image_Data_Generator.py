import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img


# Veri artırma parametrelerini belirleyin
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Görüntülerin bulunduğu klasörü belirtin
input_dir = "Train_Cropped\cropped_images_sagittal"  # Görüntülerin bulunduğu klasör
output_dir = "Augmented_Images"  # Artırılmış görüntülerin kaydedileceği klasör

# Klasörü oluştur
os.makedirs(output_dir, exist_ok=True)

# Her görüntü için veri artırma yap ve kaydet
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(
        ".png"
    ):  # Uygun uzantıları kontrol edin
        img_path = os.path.join(input_dir, filename)
        img = load_img(img_path)  # Görüntüyü yükleyin
        x = img_to_array(img)  # Görüntüyü numpy array'e çevirin
        x = x.reshape((1,) + x.shape)  # 4 boyutlu hale getirin

        # Veri artırma işlemi
        i = 0
        for batch in datagen.flow(
            x,
            batch_size=1,
            save_to_dir=output_dir,
            save_prefix="aug",
            save_format="jpg",
        ):
            i += 1
            if (
                i >= 4
            ):  # Her görüntüden 4 tane daha üretin (toplamda 5 katına çıkarmak için)
                break
