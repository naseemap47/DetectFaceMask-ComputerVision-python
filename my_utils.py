import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
import os


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