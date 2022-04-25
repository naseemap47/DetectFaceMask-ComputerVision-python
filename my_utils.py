import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np


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
        rescale=1 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    val_preprocessor = ImageDataGenerator(rescale=1 / 255)
    test_preprocessor = ImageDataGenerator(rescale=1 / 255)
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


def predict_with_model(img_path, model):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [70, 70])  # (70,70,3)
    image = tf.expand_dims(image, axis=0)  # (1,70,70,3)

    # Predict
    predictions = model.predict(image)  # [0.009,0.09,0.99, 0.0009,...]
    predictions = np.argmax(predictions)  # 3

    if predictions == 0:
        predictions = 'WithMask'
    else:
        predictions = 'WithoutMask'
    return predictions
