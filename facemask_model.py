from my_utils import sample_images
from my_utils import create_generators

Sample = False

if Sample:
    path_img = '/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/FaceMaskDataset/Test/WithMask'
    sample_images(path_img)

train_generators, val_generators, test_generators = create_generators(
    batch_size=32,
    path_to_train_data='/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/FaceMaskDataset/Train',
    path_to_val_data='/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/FaceMaskDataset/Validation',
    path_to_test_data='/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/FaceMaskDataset/Test'
)
