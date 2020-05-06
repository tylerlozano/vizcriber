import math
import os
import random
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from .data_loader import DataLoader
from .models import (create_attention, create_decoder, create_embedder,
                     create_encoder)


class ModelTrainer():
    """
    The ModelTrainer class instantiates, trains, and loads the encoder-decoder
    architecture defined in the models module. Pass training=False to just
    instantiate the base model or load a checkpoint.

    Methods
    -------
    load_model(target=None)
        Either loads latest checkpoint according to the checkpoint manager
        or a specified checkpoint indicated by target

    train(epochs)
        Trains model iterating over training dataset epochs number of times

    unfreeze_embeddings
        Unfreezes embeddings in decoder model so they become trainable

    unfreeze_encoder(freeze_pt=86)
        Unfreezes layers in encoder from freeze_pt layer to the last layer
    """
    _scc = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def __init__(self, expected_total_epochs=35, vocab_size=10240,
                 embed_dim=512, dec_units=512, att_units=512,
                 learning_rate=0.0007, training=True):
        """
        Parameters
        ----------
        expected_total_epochs : int, optional
            number of total epochs intended to train model over, which helps
            flatten the inverse sigmoid curve for scheduled sampling
        vocab_size : int, optional
            number of words to use in model from original MSCOCO corpus
            gathered from captions; affects model accuracy and memory used
        embed_dim : int, optional
            size of embedding layer to embed the feature vector from encoder
        dec_units : int, optional
            size of gated recurrent unit layer in decoder
        att_units : int, optional
            size of 2 dense layers inside attention model
        learning_rate : float, optional
            degree of learning applied by Adam optimizer
        training : bool
            whether ModelTrainer is going to be used to train or to help
            execute models; if train=False only load_model should be used
        """
        self.data_loader = DataLoader(vocab_size=vocab_size, training=training)

        encoder = create_encoder(Xception)
        encoder.trainable = False
        enc_out_shape = (10, 10, 2048)

        embedder = create_embedder(enc_out_shape, embed_dim)
        self.emb_shape = (100, embed_dim)

        cvec_shape = (self.emb_shape[0])
        if training:
            decoder = create_decoder(cvec_shape,
                                     self.data_loader.vocab_size + 1,
                                     self.data_loader.WORD_EMBED_SIZE,
                                     dec_units,
                                     pretrained_embeddings=(
                                         self.data_loader.
                                         create_word_embedding_matrix())
                                     )
        else:
            decoder = create_decoder(cvec_shape,
                                     self.data_loader.vocab_size + 1,
                                     self.data_loader.WORD_EMBED_SIZE,
                                     dec_units
                                     )
        self.dec_units = dec_units
        dec_state_shape = (dec_units)

        attention = create_attention(
            dec_state_shape, self.emb_shape, att_units)

        optimizer = Adam(learning_rate)

        if training:
            from .evaluator import Evaluator
            self.decay = self._create_decay_scheduler(self._inverse_sigmoid,
                                                      expected_total_epochs)
            self.evaluator = Evaluator()
        
        self.ckpt = self._init_checkpoint(
            encoder, embedder, attention, decoder, optimizer, training)
        self.ckpt_path = f"{os.path.abspath('.')}/training/models/model_checkpoints"
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=self.ckpt_path, max_to_keep=None)

    # to-do handle error
    def load_model(self, target=None):
        """Either loads latest checkpoint determined by checkpoint manager or
        target checkpoint passed by name

        Parameters
        ----------
        target : string, optional (default None)
            Name of checkpoint to be loaded"""
        if target is not None:
            ckpt_path = f"{self.ckpt_path}/{target}"
        elif self.ckpt_manager.latest_checkpoint:
            ckpt_path = self.ckpt_manager.latest_checkpoint
        self.ckpt.restore(ckpt_path)

    # todo :
    # potential bug with restarts on initial epoch without latest_checkpoint
    # refactor into train and validate
    def train(self, epochs):
        """Trains model for given number of epochs

        Parameters
        ----------
        epochs : int
            Number of epochs to train model"""
        start_epoch = 0
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            start_epoch = self.ckpt.epoch.numpy()

        train_batches = self.data_loader.get_train_dataset()
        val_batches = self.data_loader.get_val_dataset()

        # remainder batches less than batch size are dropped
        train_batches_per_epoch = math.floor(self.data_loader.TRAIN_SIZE /
                                             self.data_loader.batch_size)
        val_batches_per_epoch = math.floor(
            self.data_loader.VAL_SIZE / self.data_loader.batch_size)

        for epoch in range(start_epoch + 1, (start_epoch + 1) + epochs):
            # start of epoch
            # start of training
            tr_loss, va_loss = 0, 0
            finished_batches = train_batches_per_epoch * (epoch - 1)
            for (batch, (images, captions)) in enumerate(train_batches):
                _loss = self._train_step(images, captions,
                                         batch + finished_batches)
                tr_loss += _loss

                if batch % 100 == 0:
                    print(
                        f"Epoch {epoch} Batch {batch} Loss {tr_loss / (batch + 1)}")
            # end of training
            tr_loss /= train_batches_per_epoch
            self.ckpt.train_loss.assign(tr_loss)
            tr_loss = 0

            # start of validation
            self.evaluator.reset_scores()
            for (batch, (images, references)) in enumerate(val_batches):
                hypotheses, _loss = self._evaluate_images(images, references)
                va_loss += _loss
                try:
                    self.evaluator.update_scores(
                        self.data_loader.batch_convert_to_words(references),
                        self.data_loader.batch_convert_to_words(hypotheses,
                                                                references=False))
                # end of validation
                except Exception as e:
                    print(
                        f'{repr(e)}| likely broken pipe in evaluator, reset.')
            va_loss /= val_batches_per_epoch
            print(f"Validation Loss: {va_loss}")
            self.ckpt.validation_loss.assign(va_loss)
            va_loss = 0

            self.evaluator.set_ckpt_scores(self.ckpt)
            self.ckpt.epoch.assign(epoch)

            self.ckpt_manager.save()

            print(self.ckpt.scores)

    def unfreeze_embeddings(self):
        """Unfreezes embeddings in decoder model so they become trainable"""
        self.ckpt.decoder.layers[2].trainable = True
        self.ckpt_manager.save()

    def unfreeze_encoder(self, freeze_pt=86):
        """Unfreezes layers in encoder from freeze_pt to the last layer"""
        last_layer = len(self.ckpt.encoder.layers)
        for i in range(freeze_pt, last_layer):
            self.ckpt.encoder.layers[i].trainable = True
        self.ckpt_manager.save()

    def _train_step(self, images, captions, batch_number):
        # trains model over single batch
        total_scc_loss = 0
        sentence_length = captions.shape[1]
        attention_weights_list = np.zeros(
            (self.data_loader.batch_size, sentence_length, self.emb_shape[0]))  # fix this
        word = tf.expand_dims(
            [self.data_loader.tokenizer.word_index['<start>']] * self.data_loader.batch_size, 1)
        decoder_state = tf.zeros((self.data_loader.batch_size, self.dec_units))

        with tf.GradientTape() as main_tape:
            enc_out = self.ckpt.encoder(images)
            features = self.ckpt.embedder(enc_out)

            for i in range(1, sentence_length):
                context_vector, attention_weights = self.ckpt.attention(
                    [decoder_state, features])
                word_logits, decoder_state = self.ckpt.decoder(
                    [word, context_vector])
                attention_weights_list[:, i, :] = attention_weights

                if self._choose(self.decay, batch_number):
                    word = captions[:, i]
                else:
                    word = self._sample_words(word_logits, k=5, temp=(0.06))

                word = tf.expand_dims(word, 1)

                total_scc_loss += self._scc_loss(captions[:, i], word_logits)

            # calculate in context to track gradients
            ds_loss = self._dsar_loss(attention_weights_list)
            loss = ds_loss + total_scc_loss

        all_trainable_variables = self.ckpt.encoder.trainable_variables + \
            self.ckpt.embedder.trainable_variables + \
            self.ckpt.decoder.trainable_variables + \
            self.ckpt.attention.trainable_variables

        all_gradients = main_tape.gradient(loss, all_trainable_variables)

        self.ckpt.optimizer.apply_gradients(
            zip(all_gradients, all_trainable_variables))
        normalized_scc_loss = total_scc_loss / sentence_length

        return normalized_scc_loss

    def _evaluate_images(self, images, references):
        # evaluates loss of images over references
        total_scc_loss = 0
        sampled_captions = self.data_loader.batch_sample_target(references)
        sentence_length = sampled_captions.shape[1]
        word = tf.expand_dims(
            [self.data_loader.tokenizer.word_index['<start>']] * images.shape[0], 1)
        decoder_state = tf.zeros((images.shape[0], self.dec_units))
        features = self.ckpt.embedder(self.ckpt.encoder(images))
        pad = self.data_loader.tokenizer.word_index['<pad>']
        end = self.data_loader.tokenizer.word_index['<end>']
        start = self.data_loader.tokenizer.word_index['<start>']
        sentences = [[start] * images.shape[0]]

        # while <end> token not predicted
        for i in range(1, sentence_length):

            context_vector, _ = self.ckpt.attention([decoder_state, features])
            word_logits, decoder_state = self.ckpt.decoder(
                [word, context_vector])

            word = self._sample_words(word_logits, k=2, temp=0.0001)

            sentences.append(word.numpy())

            word = tf.expand_dims(word, 1)

            total_scc_loss += self._scc_loss(
                sampled_captions[:, i], word_logits)

        normalized_scc_loss = total_scc_loss / sentence_length

        last_word = []
        for i in sentences[-1]:
            if i != end:
                last_word.append(end)
            else:
                last_word.append(pad)
        sentences.append(last_word)

        _sentences = []
        for j in range(images.shape[0]):
            sentence = [sentences[i][j] for i in range(len(sentences))]
            _sentences.append(sentence)

        sentences = tf.convert_to_tensor(_sentences)

        return sentences, normalized_scc_loss

    def _init_checkpoint(self, encoder, embedder, attention, decoder, optimizer, training):
        # initializes checkpoint
        ckpt = tf.train.Checkpoint(epoch=tf.Variable(0, dtype='int8'),
                                   train_loss=tf.Variable(0, dtype="float32"),
                                   validation_loss=tf.Variable(
                                       0, dtype="float32"),
                                   encoder=encoder,
                                   embedder=embedder,
                                   decoder=decoder,
                                   attention=attention,
                                   optimizer=optimizer)
        if training:
            scores = {k: tf.Variable(v, dtype='float32')
            for k, v in self.evaluator.get_scores().items()}
        else:
            scores = {}
        ckpt.scores = scores
        return ckpt

    def _create_decay_scheduler(self, function, expected_total_epochs):
        # flattens decay function (inverse sigmoid) over expected_total_epochs
        num_batches = math.ceil(
            (self.data_loader.DATASET_SIZE * self.data_loader.TRAIN_P) / self.data_loader.batch_size)
        k_ = num_batches * (1 + (expected_total_epochs / 25))
        return partial(function, k=k_)

    def _inverse_sigmoid(self, x, k=1):
        # inverse sigmoid function
        return k / (k + math.exp(x / k))

    def _choose(self, scheduler, ith_batch):
        # True for teacher forcing, False for sampling from model
        return random.random() < scheduler(ith_batch)

    @classmethod
    def _sample_words(cls, word_logits, k=100, temp=1.0):
        # top-k temperature sampling
        #
        # word_logits: (batch, vocab_size)
        # k: k largest logit scores
        # temp: float, when > 1 makes dist more uniform, when < 1 makes more ununiform
        word_logits = tf.squeeze(word_logits)
        # get k largest logits and their indices
        topk = tf.math.top_k(word_logits, k)
        # sample 1 index from each set of k indices based on logit values in indices
        sample = tf.random.categorical((topk.values / temp), 1)
        # get words associated to each index value
        sample_indices = [[i, n] for i, [n] in enumerate(sample.numpy())]
        return tf.gather_nd(topk.indices, sample_indices)

    @classmethod
    def _scc_loss(cls, y, y_hat):
        # scc loss over actual caption y and predicted caption y_hat
        y = tf.cast(y, tf.float32)
        mask = tf.math.logical_not(tf.math.equal(y, 0))
        loss = cls._scc(y, y_hat)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.reduce_mean(loss)

    @classmethod
    def _dsar_loss(cls, att_weights):
        # doubly stochastic attention regularization
        # takes attention weights across entire sequence and calculates loss
        # based on how well each pixel adds up to 1 across entire sequence

        # att_weights: float tensor of shape (batch_size, sequence_length,
        # num_pixels)
        pixels_sum_over_seq = tf.reduce_sum(
            att_weights, axis=1)  # axis-1 is time axis
        pixel_losses = ((1. - pixels_sum_over_seq) ** 2)
        total_loss = tf.cast(tf.reduce_mean(pixel_losses), tf.float32)
        return total_loss
