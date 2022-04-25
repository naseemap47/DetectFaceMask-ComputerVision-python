import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
import os
from keras.preprocessing.image import ImageDataGenerator


def sample_images(img_folder_path):
    plt.figure(figsize=(5, 5))
    for i in range(1, 10, 1):
        plt.subplot(3, 3, i)
        img = load_img(
            img_folder_path + '/' + os.listdir(img_folder_path)[i],
            target_size=(70, 70)
        )
        plt.imshow(img)
        plt.tight_layout()
        plt.axis('off')
    plt.show()


def create_generators(batch_size, path_to_train_data, path_to_val_data, path_to_test_data):
    train_preprocessor = ImageDataGenerator(
        rescale=1/255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    val_preprocessor = ImageDataGenerator(rescale=1/255)
    test_preprocessor = ImageDataGenerator(rescale=1/255)
    train_generators = train_preprocessor.flow_from_directory(
        path_to_train_data,
        target_size=(70, 70),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )
    val_generators = val_preprocessor.flow_from_directory(
        path_to_val_data,
        target_size=(70, 70),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )
    test_generators = test_preprocessor.flow_from_directory(
        path_to_test_data,
        target_size=(70, 70),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )
    return train_generators, val_generators, test_generators