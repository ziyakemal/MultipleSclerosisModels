import os
import shutil
import random

def split_dataset_per_folder(parent_folder, test_ratio=0.2):
    # Ana klasör içindeki her bir alt klasör için işlemi gerçekleştir
    for sub_folder in os.listdir(parent_folder):
        full_sub_folder_path = os.path.join(parent_folder, sub_folder)
        
        if os.path.isdir(full_sub_folder_path):
            images = [f for f in os.listdir(full_sub_folder_path) if os.path.isfile(os.path.join(full_sub_folder_path, f))]
            
            # Görüntüleri rastgele karıştır
            random.shuffle(images)
            
            # Eğitim ve test veri seti için indeks ayarla
            test_size = int(len(images) * test_ratio)
            train_images = images[test_size:]
            test_images = images[:test_size]
            
            # Eğitim ve test klasörlerini oluştur
            train_folder = os.path.join(full_sub_folder_path, 'train')
            test_folder = os.path.join(full_sub_folder_path, 'test')
            
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)
            else:
                shutil.rmtree(train_folder)
                os.makedirs(train_folder)
            
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)
            else:
                shutil.rmtree(test_folder)
                os.makedirs(test_folder)
            
            # Eğitim görüntülerini kopyala
            for image in train_images:
                shutil.copy(os.path.join(full_sub_folder_path, image), train_folder)
            
            # Test görüntülerini kopyala
            for image in test_images:
                shutil.copy(os.path.join(full_sub_folder_path, image), test_folder)
            
            print(f"{sub_folder} - Eğitim seti boyutu: {len(train_images)}")
            print(f"{sub_folder} - Test seti boyutu: {len(test_images)}")

# Kullanım
parent_folder = 'C:/Users/ziyak/Desktop/Makale/9_Poster/DataSet/Multiple Sclerosis'  # Ana klasör yolu
split_dataset_per_folder(parent_folder, test_ratio=0.25)




# import os
# import shutil
# import random

# def split_dataset(image_folder, train_folder, test_folder, test_ratio=0.2):
#     # Tüm görüntü dosyalarını al
#     images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
#     # Görüntüleri rastgele karıştır
#     random.shuffle(images)
    
#     # Eğitim ve test veri seti için indeks ayarla
#     test_size = int(len(images) * test_ratio)
#     train_images = images[test_size:]
#     test_images = images[:test_size]
    
#     # Eğitim klasörünü oluştur veya temizle
#     if not os.path.exists(train_folder):
#         os.makedirs(train_folder)
#     else:
#         shutil.rmtree(train_folder)
#         os.makedirs(train_folder)

#     # Test klasörünü oluştur veya temizle
#     if not os.path.exists(test_folder):
#         os.makedirs(test_folder)
#     else:
#         shutil.rmtree(test_folder)
#         os.makedirs(test_folder)
    
#     # Eğitim görüntülerini kopyala
#     for image in train_images:
#         shutil.copy(os.path.join(image_folder, image), train_folder)
    
#     # Test görüntülerini kopyala
#     for image in test_images:
#         shutil.copy(os.path.join(image_folder, image), test_folder)

#     print(f"Eğitim seti boyutu: {len(train_images)}")
#     print(f"Test seti boyutu: {len(test_images)}")

# # Kullanım
# image_folder = 'C:/Users/ziyak/Desktop/Makale/9_Poster/DataSet/Multiple Sclerosis'  # Orijinal görüntülerin bulunduğu klasör
# train_folder = 'C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/DataSet/Test'   # Eğitim seti için hedef klasör
# test_folder = 'C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/DataSet/Train'     # Test seti için hedef klasör

# split_dataset(image_folder, train_folder, test_folder, test_ratio=0.25)
