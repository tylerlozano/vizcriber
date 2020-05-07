from __future__ import absolute_import

import glob
import os
import re
import shutil

import tensorflow as tf

from models.data_loader import DataLoader

_URIS = {
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'validation_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'glove_embeddings': 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
}


def download_all():
    "Downloads annotations and images from MSCOCO and 300 dim GLOVE embeddings"
    download_annotations()
    download_images()
    download_glove_embeddings()


def download_annotations():
    "Downloads annotation data from MSCOCO"
    if not os.path.exists(DataLoader.ANNOTATIONS_PATH):
        annotation_zip = tf.keras.utils.get_file(
            f"{os.path.dirname(DataLoader.ANNOTATIONS_PATH)}/captions.zip",
            cache_subdir=os.path.abspath('.'),
            origin=_URIS['annotations'],
            extract=True)
        os.remove(annotation_zip)
        for f in os.listdir(DataLoader.ANNOTATIONS_PATH):
            if not re.search('^cap', f):
                os.remove(os.path.join(DataLoader.ANNOTATIONS_PATH, f))


def download_images():
    "Downloads images data from MSCOCO"
    if not os.path.exists(DataLoader.IMAGES_PATH):

        train_image_zip = tf.keras.utils.get_file(
            f"{os.path.dirname(DataLoader.IMAGES_PATH)}/train2017.zip",
            cache_subdir=os.path.abspath('.'),
            origin=_URIS['train_images'],
            extract=True)
        os.rename(f'{os.path.dirname(train_image_zip)}/train2017/',
                  DataLoader.IMAGES_PATH)
        os.remove(train_image_zip)

        val_image_zip = tf.keras.utils.get_file(
            f"{os.path.dirname(DataLoader.IMAGES_PATH)}/val2017.zip",
            cache_subdir=os.path.abspath('.'),
            origin=_URIS['validation_images'],
            extract=True)
        for image in glob.glob(f'{os.path.dirname(train_image_zip)}/val2017/*.jpg'):
            shutil.move(image, DataLoader.IMAGES_PATH)
        os.remove(val_image_zip)
        os.rmdir(f'{os.path.dirname(val_image_zip)}/val2017')


def download_glove_embeddings():
    "Downloads 300 dimensional GLOVE embeddings file"
    if not os.path.exists(DataLoader.EMBEDDINGS_PATH):
        glove_zip = tf.keras.utils.get_file(
            f"{os.path.dirname(DataLoader.EMBEDDINGS_PATH)}/glove.42B.300d.zip",
            cache_subdir=os.path.abspath('.'),
            origin=_URIS['glove_embeddings'],
            extract=True)
        os.remove(glove_zip)


download_all()
