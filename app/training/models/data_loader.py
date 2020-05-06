
import contextlib
import json
import os
import random
from collections import OrderedDict
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer


class DataLoader():
    """
    A class used to package related data processing constants and methods
    for training and executing an encoder-decoder model. It can be initialized with training=False
    to simpy get the initialized constants and word_list.json used for
    beam_search in the caption module.

    Attributes
    ----------
    WORD_EMBED_SIZE : int
        size of word embedding in decoder model
    IMAGE_SIZE : int
        size of length and width dimension for encoder input
    DATA_GEN_ARGS : dict
        a dictionary of image transformations to pass to ImageDataGenerator
    AUTOTUNE : obj
        lazy declaration of parameters and optimizes dataprocessing parameters
        when datasets are consumed, i.e. PARALLEL_CALLS and BUFFER_SIZE
    PARALLEL_CALLS : int
        number of paralllel calls to make when mapping over a batch
    BUFFER_SIZE: int
        number of batches to prefetch
    TRAIN_P, VAL_P, TEST_P: int
        percentage splits of dataset, should all add up to 1


    Methods
    -------
    get_train_dataset, get_val_dataset, get_test_dataset
        returns dataset to be consumed by training, validating, and testing
        processes respectively in ModelTrainer class

    batch_convert_to_words(captions, references=True)
        converts references and hypotheses from int sequences to words

    batch_sample_target(target)
        selects random caption from each list in target

    load(img, path=False)
        loads image from image_id or path img and resizes and converts to array

    preprocess_input(x), unprocess_input(x)
        transforms and reverses transform image tensor/array x respectively
        using transformations expected by encoder

    create_word_embedding_matrix
        creates glove embeddings of size WORD_EMBED_SIZE, must download correct
        glove embedding file to resize to other dimensions
    """

    WORD_EMBED_SIZE = 300
    IMAGE_SIZE = 299
    DATA_GEN_ARGS = dict(rotation_range=45,
                         shear_range=0.20,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=False,
                         )
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    PARALLEL_CALLS = AUTOTUNE
    BUFFER_SIZE = AUTOTUNE
    TRAIN_P, VAL_P, TEST_P = 0.9, 0.05, 0.05

    WORDS_PATH = f"{os.path.abspath('.')}/training/data/word_list.json"
    ANNOTATIONS_PATH = f"{os.path.abspath('.')}/training/data/annotations"
    IMAGES_PATH = f"{os.path.abspath('.')}/training/data/images"
    EMBEDDINGS_PATH = f"{os.path.abspath('.')}/training/data/glove.42B.300d.txt"

    def __init__(self, batch_size=32, vocab_size=10240, shuffle_size=8000, training=True):
        """
        Parameters
        ----------
        batch_size : int, optional
            number of image/caption pairs to batch from dataset; reduce if oom
        vocab_size : int, optional
            number of words to use in model from original MSCOCO corpus
            gathered from captions
        shuffle_size : int, optional
            how big the shuffle window is when shuffling the dataset
        training : bool
            whether DataLoader is going to be used to train or to help
            execute models; if train=False non-class methods should not be used
        """
        assert (vocab_size > 3 and
                vocab_size < 28040), "vocab_size not in range"
        self.vocab_size = vocab_size
        if (not os.path.exists(self.WORDS_PATH) or training):
            coco_train_file = f'{self.ANNOTATIONS_PATH}/captions_train2017.json'
            coco_val_file = f'{self.ANNOTATIONS_PATH}/captions_val2017.json'
            coco_files = [coco_train_file, coco_val_file]

            img_to_caps = OrderedDict()

            with contextlib.ExitStack() as stack:
                files = [stack.enter_context(open(fname, 'r'))
                         for fname in coco_files]
                for f in files:
                    self._coco(f, img_to_caps)

            self.tokenizer = Tokenizer(num_words=self.vocab_size,
                                       oov_token="<unk>",
                                       filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
            self.tokenizer.fit_on_texts(tf.nest.flatten(img_to_caps.values()))
            self.tokenizer.index_word[0] = '<pad>'
            self.tokenizer.word_index['<pad>'] = 0

            if not os.path.exists(self.WORDS_PATH):
                self._create_word_list(self.WORDS_PATH)

        if training:
            # set instance variables
            self.DATASET_SIZE = len(img_to_caps.values())
            self.batch_size = batch_size
            self.shuffle_size = shuffle_size

            self.image_datagen = ImageDataGenerator(**self.DATA_GEN_ARGS)

            # set functions
            self.batch_load = partial(tf.map_fn, self.load, dtype='float32')
            self.batch_transform = partial(tf.map_fn, self._transform,
                                           dtype='float32')
            self.batch_preprocess = partial(tf.map_fn, self._preprocess,
                                            dtype='float32')
            # captions list
            captions = self._process_captions(list(img_to_caps.values()))
            # data splits
            self.train, self.val, self.test = self._split_data(list(
                img_to_caps.keys()), captions)

    def _create_word_list(self, path):
        # create word_list at indicated path
        data = []
        for i in range(self.vocab_size + 1):
            data.append(self.tokenizer.index_word[i])
        with open(path, 'w') as p:
            json.dump(data, p, indent=4)

    def _train_mapper(self, paths, seq_lists):
        # wrapper over _map_train_batch to map over tensors from train dataset
        return tf.numpy_function(self._map_train_batch, [paths, seq_lists],
                                 [tf.float32, tf.int32])

    def _validate_mapper(self, paths, seqs):
        # wrapper over _map_validate_batch to map over tensors from val dataset
        return tf.numpy_function(self._map_validate_batch, [paths, seqs],
                                 [tf.float32, tf.int32])

    def _transform(self, img):
        # wrapper over random_transform to be used in tensor mapping
        return tf.numpy_function(self.image_datagen.random_transform,
                                 [img], [tf.float32])

    def _preprocess(self, img):
        # wrapper over preprocess_input to be used in tensor mapping
        return tf.numpy_function(self.preprocess_input, [img],
                                 [tf.float32])

    def _map_train_batch(self, paths, captions):
        # core map function over train batches
        imgs = self.batch_load(paths)
        imgs = self.batch_transform(imgs)
        imgs = self.batch_preprocess(imgs)
        captions = self.batch_sample_target(captions)
        return imgs, captions

    def _map_validate_batch(self, paths, captions):
        # core map function over validate batches
        imgs = self.batch_load(paths)
        imgs = self.batch_preprocess(imgs)
        return imgs, captions

    def _split_data(self, x_list, y_list):
        # combines each member from x_list and y_list into an (x, y)
        # tuple and splits the resulting list into train, validate, and test
        # sublists to be later converted into datasets
        train_split = int(self.TRAIN_P * self.DATASET_SIZE)
        val_split = int((self.TRAIN_P + self.VAL_P) * self.DATASET_SIZE)

        dataset_list = list(zip(x_list, y_list))
        # don't shuffle data before split, need inherent ordering
        # to train after restarting DataLoader
        train, validate, test = np.split(dataset_list,
                                         [train_split, val_split])
        return train, validate, test

    def _make_datasets(self, xy_list):
        # converts a list of (x, y) tuples into two separate datasets of
        # x and y respectively
        images = [x for x, y in xy_list]
        captions = [y for x, y in xy_list]
        images_ds = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(images, dtype='int32')))
        captions_ds = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(captions, dtype='int32')))
        return images_ds, captions_ds

    def _coco(self, json_file, d):
        # load json data into ordered dictionary of image_id : captions pairs
        annotations = json.load(json_file)
        for annot in annotations['annotations']:
            path = int(annot['image_id'])
            l = d.get(path, [])
            l.append(f"<start> {annot['caption']} <end>")
            d[path] = l

    def _process_captions(self, captions):
        # preprocesses captions by tokenizing, cropping, padding, and removing
        # any caption list with more than 5 captions, prefering to remove
        # shorter captions

        min_len, max_len = 10, 18
        end = self.tokenizer.word_index['<end>']

        # return False if sequence is not of acceptable length
        def F(s): return min_len <= len(s) <= max_len

        # pad sequences up to max length with <pad>
        def P(s): return tf.keras.preprocessing.sequence.pad_sequences(
            s, padding='post', maxlen=max_len)

        # return number of sequences over max allowed of 5
        def R(s): return len(s) - 5

        # returns indices of R sequences with least number of words
        def Z(captions):
            over = len(captions) - 5
            size_list = [len(c) for c in captions]
            res = sorted(range(len(size_list)),
                         key=lambda sub: size_list[sub])[:over]
            return res

        # tokenize captions
        captions = [self.tokenizer.texts_to_sequences(
            cap_list) for cap_list in captions]

        # crop sequences of incorrect length
        captions = [[cap if F(cap) else cap[:max_len - 1] + [end]
                     for cap in cap_list] for cap_list in captions]

        # remove cap_lists that have length > 5
        captions = [[c for i, c in enumerate(c_l) if i not in Z(
            c_l)] if R(c_l) else c_l for c_l in captions]

        # pad sequences to length max-length
        captions = [P(cap_list) for cap_list in captions]

        return captions

    def get_train_dataset(self):
        """Returns training dataset optimized to be consumed batchwise over
        many epochs
        """
        train_dataset = (
            tf.data.Dataset.zip((self._make_datasets(self.train)))
            .cache()
            .shuffle(self.shuffle_size)
            .batch(self.batch_size, drop_remainder=True)
            .map(self._train_mapper, num_parallel_calls=self.PARALLEL_CALLS)
            .prefetch(self.BUFFER_SIZE))

        return train_dataset

    def get_val_dataset(self):
        """Returns validation dataset optimized to be consumed batchwise over
        many epochs
        """
        val_dataset = (
            tf.data.Dataset.zip((self._make_datasets(self.val)))
            .cache()
            .shuffle(self.shuffle_size)
            .batch(self.batch_size, drop_remainder=True)
            .map(self._validate_mapper, num_parallel_calls=self.PARALLEL_CALLS)
            .prefetch(self.BUFFER_SIZE)
        )
        return val_dataset

    def get_test_dataset(self):
        """Returns test dataset optimized to be consumed batchwise over
        a single epoch
        """
        test_dataset = (
            tf.data.Dataset.zip((self._make_datasets(self.test)))
            .batch(self.batch_size, drop_remainder=True)
            .map(self._validate_mapper, num_parallel_calls=self.PARALLEL_CALLS)
            .prefetch(self.BUFFER_SIZE)
        )
        return test_dataset

    def batch_convert_to_words(self, captions, references=True):
        """Converts integer tensor to array of strings with same nested 
        structure except not a tensor and converted from int32 to tokens from 
        tokenizer with <start> and <end> tokens removed

        Parameters
        ----------
        captions : tensor of int
            either a tensor of shape (batch, list_size, padded_sequence_length)
            for references or (batch, padded_sequence_length) for hypotheses"""
        captions = captions.numpy()
        # tokenize sequences
        T = self.tokenizer.sequences_to_texts
        end = self.tokenizer.word_index['<end>']
        # find where the sequence ends with <end> token
        def E(caption): return np.where(caption == end)[0][0]

        sentences = []
        if references:
            filtered = [[sentence[1:E(sentence)]
                         for sentence in lst] for lst in captions]
            sentences = [T(lst) for lst in filtered]
        else:
            filtered = [sentence[1:E(sentence)] for sentence in captions]
            sentences = T(filtered)
        return sentences

    @classmethod
    def batch_sample_target(cls, target):
        """Returns a tensor of randomly selected sequences, from each list, of
        size (batch, padded_sequence_length)

        Parameters
        ----------
        target : int tensor of shape (batch, list_size, padded_sequence_length)
        """
        list_size = target.shape[1]  # should be 5
        return tf.cast([sample[np.random.randint(list_size)] for
                        sample in target], dtype='int32')

    @classmethod
    def load(cls, img, path=False):
        """Loads image and resizes / converts to array

        Parameters
        ----------
        img : str -- image_id or path to image, specified with path parameter
        path : bool -- indicates whether img is a path to an image"""
        if not path:
            img = f"{cls.IMAGES_PATH}/{img:012d}.jpg"
        img = image.load_img(img, target_size=(cls.IMAGE_SIZE, cls.IMAGE_SIZE))
        img = image.img_to_array(img)
        return img

    @classmethod
    def preprocess_input(cls, x):
        """transforms image tensor/array x using transformations expected by 
        encoder

        Parameters
        ----------
        x : image array/tensor"""
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    @classmethod
    def unprocess_input(cls, x):
        """undoes transform image tensor/array x using transformations expected by 
        encoder

        Parameters
        ----------
        x : transformed image array/tensor"""
        x /= 2.
        x += 0.5
        x *= 255.
        return x

    def create_word_embedding_matrix(self):
        """creates glove embeddings of size WORD_EMBED_SIZE, must download 
        correct glove embedding file to resize to other dimensions"""
        word_embed_dict = dict()

        with open(self.EMBEDDINGS_PATH, 'r') as file:
            for line in file:
                values = line.split()
                word = values[0]
                # only map words that are in model vocabulary
                if self.tokenizer.word_index.get(word, self.vocab_size + 1) <= self.vocab_size:
                    embedding = np.asarray(values[1:], dtype='float32')
                    word_embed_dict[word] = embedding

        index_embed_matrix = np.zeros(
            (self.vocab_size + 1, self.WORD_EMBED_SIZE))
        for i in range(self.vocab_size + 1):
            w = self.tokenizer.index_word[i]
            word_embedding = word_embed_dict.get(w)
            if word_embedding is not None:
                index_embed_matrix[i] = word_embedding

        return index_embed_matrix
