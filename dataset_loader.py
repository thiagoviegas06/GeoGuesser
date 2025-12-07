import os
import numpy as np


#image_000000_east.jpg
def load_dataset(dataset_path):
    #group east west north south

    images_dict = {}
    for image_file in os.listdir(dataset_path):
        img_number = str(image_file).split('_')[1]
        print(f"Number {img_number}")
        direction = str(image_file).split('_')[2].split('.')[0]
        print(f"Direction {direction}")
        if img_number not in images_dict:
            images_dict[img_number] = {}
        images_dict[img_number][direction] = os.path.join(dataset_path, image_file)
    return images_dict

def split_dataset(images_dict, train_ratio=0.8):
    image_numbers = list(images_dict.keys())

    # Random shuffle of indices
    shuffled_indices = np.random.permutation(len(image_numbers))
    shuffled_keys = [image_numbers[i] for i in shuffled_indices]

    split_index = int(len(shuffled_keys) * train_ratio)
    train_keys = shuffled_keys[:split_index]
    val_keys = shuffled_keys[split_index:]

    train_dict = {key: images_dict[key] for key in train_keys}
    val_dict = {key: images_dict[key] for key in val_keys}

    return train_dict, val_dict