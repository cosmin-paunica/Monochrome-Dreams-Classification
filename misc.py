import numpy as np
import imageio


NUM_TRAIN_IMAGES = 30001
NUM_VALIDATION_IMAGES = 5000
NUM_TEST_IMAGES = 5000
IMG_WIDTH, IMG_HEIGHT = 32, 32
NUM_CLASSES = 9


# citirea imaginilor
# se va apela cu read_labels==True pentru imaginile de antrenare si de validare
# se va apela cu read_labels==False pentru imaginile de testare
# for_CNN==True va face ca imaginile sa fie citite sub forma unor array-uri tridimensionale (2 dimensiuni ale imaginii si una pentru canalele de culori)
# for_CNN==False va face ca imaginile sa fie citite sub forma unor array-uri unidimensionale, cu pixelii unei imagini insirati unul dupa altul
def read_images(file, num_images, path, read_labels=True, for_CNN=False):
    if not for_CNN:
        images = np.zeros((num_images, IMG_WIDTH * IMG_HEIGHT))
    else:
        images = np.zeros((num_images, IMG_WIDTH, IMG_HEIGHT, 1))
    if read_labels:
        labels = np.zeros(num_images, 'int')
    with open(file) as f_in:
        for i, img_string in enumerate(f_in):
            if read_labels:
                img_info = img_string.split(',')
                img_file = img_info[0]
                labels[i] = int(img_info[1])
            else:
                img_file = img_string.split('\n')[0]
            if not for_CNN:
                images[i] = np.reshape(
                    np.asarray(imageio.imread(f'{path}/{img_file}')),
                    IMG_WIDTH * IMG_HEIGHT
                )
            else:
                images[i] = np.asarray(imageio.imread(f'{path}/{img_file}')).reshape((IMG_WIDTH, IMG_HEIGHT, 1))
        if read_labels:
            return images, labels
        else:
            return images


def read_all(for_CNN=False):
    train_images, train_labels = read_images(
        './data/train.txt',
        NUM_TRAIN_IMAGES,
        './data/train',
        for_CNN=for_CNN
    )
    validation_images, validation_labels = read_images(
        './data/validation.txt',
        NUM_VALIDATION_IMAGES,
        './data/validation',
        for_CNN=for_CNN
    )
    test_images = read_images(
        './data/test.txt',
        NUM_TEST_IMAGES,
        './data/test',
        False,
        for_CNN=for_CNN
    )
    return train_images, train_labels, validation_images, validation_labels, test_images


def write_predictions(file, predictions):
    with open(file, 'w') as f_out:
        f_out.write('id,label\n')
        for i in range(5000):
            f_out.write(f'0{35000 + i + 1}.png,{predictions[i]}\n')
