from my_utils import sample_images
from my_utils import create_generators
from keras.callbacks import EarlyStopping
from deeplearning_model import faceMask_model
import os
from keras.models import load_model

##########################
# Switches
SAMPLE = False
TRAIN = False
TEST = True
##########################

if SAMPLE:
    path_img = '/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/FaceMaskDataset/Test/WithMask'
    sample_images(path_img)

train_generators, val_generators, test_generators = create_generators(
    batch_size=32,
    path_to_train_data='/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/FaceMaskDataset/Train',
    path_to_val_data='/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/FaceMaskDataset/Validation',
    path_to_test_data='/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/FaceMaskDataset/Test'
)

early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=10,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

if TRAIN:
    model = faceMask_model(2)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_generators,
        batch_size=32,
        epochs=50,
        validation_data=val_generators,
        callbacks=[early_stopping]
    )

    # Save Model in a h5 format
    if os.path.isfile(
            '/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/Model.h5'
    ) is False:
        model.save(
            '/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/Model.h5'
        )

if TEST:
    saved_model = load_model('/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/Model.h5')
    saved_model.summary()

    # Evaluate Validation dataset
    print("Evaluate Validation data:")
    saved_model.evaluate(val_generators)

    # Evaluate Test dataset
    print("Evaluate Test data:")
    saved_model.evaluate(test_generators)