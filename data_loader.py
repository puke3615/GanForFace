from keras.preprocessing.image import ImageDataGenerator
import os

# PATH = 'G:/Dataset/CelebA/Img/img_align_celeba_png.7z/dataset/img_align_celeba_png'
PATH = 'G:/Dataset/CelebA/Img/img_align_celeba_png.7z/dataset'


def load_data(batch_size=32, height=28, width=28):
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
    return generator.flow_from_directory(PATH, target_size=(height, width), batch_size=batch_size)
    # return generator.flow_from_directory(PATH, target_size=(height, width), batch_size=batch_size, save_to_dir='images')


if __name__ == '__main__':
    generator = load_data()
    for x, y in generator:
        print(x.shape)
        print(y.shape)
