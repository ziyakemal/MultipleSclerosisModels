import os
import random


def delete_random_half_of_images(folder_path):
    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Filter only image files (assuming common image extensions)
    image_files = [
        f
        for f in all_files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"))
    ]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Determine how many files to delete (half of them)
    num_files_to_delete = len(image_files) // 4

    # Delete the selected files
    for i in range(num_files_to_delete):
        file_to_delete = os.path.join(folder_path, image_files[i])
        os.remove(file_to_delete)
        print(f"Deleted: {file_to_delete}")

    print(f"Deleted {num_files_to_delete} out of {len(image_files)} image files.")


# Example usage
folder_path = "Augmented_Images/Train/cropped_images_notumor"  # Replace with the path to your folder
delete_random_half_of_images(folder_path)
