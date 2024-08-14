import matplotlib.pyplot as plt
import os
import pandas as pd


def clean_label(label):
    # Remove 'cropped', 'image', or 'images' from the label
    return (
        label.replace("cropped_", "")
        .replace("cropped", "")
        .replace("image_", "")
        .replace("images_", "")
        .replace("image", "")
        .replace("images", "")
    )


def create_dataframe(data_path):
    filepath = []
    label = []
    image_folder = os.listdir(data_path)
    for folder in image_folder:
        folder_path = os.path.join(data_path, folder)
        filelist = os.listdir(folder_path)
        for file in filelist:
            # Clean the filename
            clean_file = clean_label(file)
            new_path = os.path.join(folder_path, clean_file)
            os.rename(os.path.join(folder_path, file), new_path)
            filepath.append(new_path)
            # Clean the folder (label) name as well
            label.append(clean_label(folder))
    image_data = pd.Series(filepath, name="image_data")
    label_data = pd.Series(label, name="label")
    df = pd.concat([image_data, label_data], axis=1)
    return df


# Data paths
# train_data = "C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/Train_Cropped"
train_data = "Augmented_Images/Train"
# test_data = "C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/Test_Cropped"
test_data = "Augmented_Images/Test"
# valid_data = "C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/Test_Cropped"
valid_data = "Augmented_Images/Test"

train_df = create_dataframe(train_data)
test_df = create_dataframe(test_data)


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
colors_train = ["seagreen", "slategrey", "dimgray", "lightgray", "rosybrown"]
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
colors_test = ["seagreen", "slategrey", "dimgray", "lightgray", "rosybrown"]
explode_test = [0.1 if label == "notumor" else 0 for label in label_counts_test.keys()]
create_pie_chart(
    axes[1], label_percentages_test, "Test Data Distribution", colors_test, explode_test
)

plt.tight_layout()
plt.savefig("Cleaned_TrainTestDataDistribution.png")
print("Saved -- > Cleaned_TrainTestDataDistribution.png")
plt.close()


# import matplotlib.pyplot as plt
# import os
# import pandas as pd


# def create_dataframe(data_path):
#     filepath = []
#     label = []
#     image_folder = os.listdir(data_path)
#     for folder in image_folder:
#         folder_path = os.path.join(data_path, folder)
#         filelist = os.listdir(folder_path)
#         for file in filelist:
#             new_path = os.path.join(folder_path, file)
#             filepath.append(new_path)
#             label.append(folder)
#     image_data = pd.Series(filepath, name="image_data")
#     label_data = pd.Series(label, name="label")
#     df = pd.concat([image_data, label_data], axis=1)
#     return df

# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # & ____________________________________ DataFramelerin Elde Edilmesi __________________________________________
# # * 4-) --------------------------------------------------------------------------------------------------------

# train_data = "C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/Train_Cropped"
# test_data = "C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/Test_Cropped"
# valid_data = "C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/Test_Cropped"

# train_df = create_dataframe(train_data)
# test_df = create_dataframe(test_data)
# # valid_df = create_dataframe(valid_data)

# def create_pie_chart(ax, data, title, colors, explode):
#     ax.pie(
#         data.values(),
#         labels=data.keys(),
#         autopct="%1.1f%%",
#         startangle=140,
#         colors=colors,
#         explode=explode,
#     )
#     ax.axis("equal")
#     ax.set_title(title, weight="bold")


# # Create subplots
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# # Plot for train_data
# label_counts_train = train_df["label"].value_counts().to_dict()
# total_count_train = sum(label_counts_train.values())
# label_percentages_train = {
#     label: count / total_count_train * 100
#     for label, count in label_counts_train.items()
# }
# # Added new color for "ms" class
# colors_train = ["seagreen", "slategrey", "dimgray", "lightgray", "rosybrown"]
# # Added explode for "ms" class
# explode_train = [
#     0.1 if label in ["notumor"] else 0 for label in label_counts_train.keys()
# ]
# create_pie_chart(
#     axes[0],
#     label_percentages_train,
#     "Train Data Distribution",
#     colors_train,
#     explode_train,
# )

# # Plot for test_data
# label_counts_test = test_df["label"].value_counts().to_dict()
# total_count_test = sum(label_counts_test.values())
# label_percentages_test = {
#     label: count / total_count_test * 100 for label, count in label_counts_test.items()
# }
# # Added new color for "ms" class
# colors_test = ["seagreen", "slategrey", "dimgray", "lightgray", "rosybrown"]
# # Added explode for "ms" class
# explode_test = [
#     0.1 if label in ["notumor"] else 0 for label in label_counts_test.keys()
# ]
# create_pie_chart(
#     axes[1], label_percentages_test, "Test Data Distribution", colors_test, explode_test
# )

# plt.tight_layout()
# plt.savefig("TrainTestDataDistribution.png")
# print("Saved -- > Train_Test_Data_Distribution.png")
# plt.close()
