from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
import os

# PATH = 'G:/Dataset/CelebA/Img/img_align_celeba_png.7z/dataset/img_align_celeba_png'
# PATH = 'G:/Dataset/CelebA/Img/img_align_celeba_png.7z/dataset3'
PATH = 'G:/Dataset/CelebA/Img/img_align_celeba_png.7z/dataset'

FACE_PATH = 'G:/Dataset/FacialKeyPointDetection'


def load_data(batch_size=32, height=28, width=28, imgaug=False):
    if not os.path.exists(PATH):
        raise Exception('File folder "%s" not found' % PATH)
    generator = ImageDataGenerator(
        # samplewise_std_normalization=True,
        # samplewise_center=True,
        # channel_shift_range=15,
        horizontal_flip=True,
        # rotation_range=15,
        # width_shift_range=.2,
        # height_shift_range=.2,
        # zoom_range=.01,
    )
    iterator = generator.flow_from_directory(
        PATH, target_size=(height, width), batch_size=batch_size)
    if imgaug:
        while True:
            batch, _ = next(iterator)
            batch = img_aug(batch)
            yield batch, _
    else:
        return iterator


def load_face_data(batch_size=32, height=28, width=28, offset=30, channel=3, imgaug=False):
    if not os.path.exists(FACE_PATH):
        raise Exception('File folder "%s" not found' % FACE_PATH)
    train_images = load_image_from_text(os.path.join(FACE_PATH, 'trainImageList.txt'))
    test_images = load_image_from_text(os.path.join(FACE_PATH, 'testImageList.txt'))
    images = [*train_images, *test_images]
    image_size = len(images)
    epoch = 0
    while True:
        epoch += 1
        sep = ''.join(['='] * 200)
        print('Epoch %d %s' % (epoch, sep))

        n_batch = image_size // batch_size
        index_list = list(range(image_size))
        for batch_index in range(n_batch):
            index_select = np.random.choice(index_list, batch_size, replace=False)
            batch_images = [images[i] for i in index_select if index_list.remove(i) or True]
            batch = np.empty((batch_size, height, width, channel), np.float32)
            for index, (image, box) in enumerate(batch_images):
                image_path = os.path.join(FACE_PATH, image)
                im = Image.open(image_path)
                left = box[0] - offset
                top = box[1] - offset
                right = box[2] + offset
                bottom = box[3] + offset
                box = [max(0, left), max(0, top), min(im.width, right), min(im.height, bottom)]
                im = im.crop(box)
                im = im.resize((width, height))
                # print(im.width, im.height)
                # im.show()
                batch[index, ...] = np.asarray(im)
            if imgaug:
                batch = img_aug(batch)
            yield batch


def load_image_from_text(text_path):
    if not os.path.exists(text_path):
        raise Exception('File "%s" not found' % text_path)
    result = []
    with open(text_path) as f:
        for line in f:
            items = line.split(' ')
            image = items[0]
            box = [int(items[1]), int(items[3]), int(items[2]), int(items[4])]
            result.append((image, box))
    return result


def img_aug(images):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-15, 15),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    return seq.augment_images(images)


if __name__ == '__main__':
    # generator = load_data()
    # for x, y in generator:
    #     print(x.shape)
    #     print(y.shape)

    data = load_face_data(3, 128, 128, offset=30, imgaug=True)
    [Image.fromarray(image).show() for image in next(data)]

    # sep = ''.join(['='] * 200)
    # print('Epoch %d %s' % (10, sep))
