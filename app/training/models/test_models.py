import pytest

from .data_loader import DataLoader
from .evaluator import Evaluator
from .model_driver import beam_search
from .model_trainer import ModelTrainer
from .models import *
import json
import os
import numpy as np
import tensorflow as tf

"""
data_loader
"""
@pytest.fixture(scope='module')
def d_l():
    return DataLoader()


@pytest.fixture(scope='module')
def train_batches(d_l):
    return d_l.get_train_dataset()


@pytest.fixture(scope='module')
def val_batches(d_l):
    return d_l.get_val_dataset()


@pytest.fixture(scope='module')
def test_batches(d_l):
    return d_l.get_test_dataset()


@pytest.mark.parametrize("vocab_size",
                         [pytest.param(3, marks=pytest.mark.xfail(
                             raises=AssertionError)),
                          pytest.param(28040, marks=pytest.mark.xfail(
                              raises=AssertionError)),
                          (4), (10240), (28039)])
@pytest.mark.parametrize("training", [True, False])
def test_data_loader(vocab_size, training):
    dl = DataLoader(vocab_size=vocab_size, training=training)
    assert dl.vocab_size == vocab_size


def test_create_word_list(d_l, tmp_path):
    directory = tmp_path / "sub"
    directory.mkdir()
    path = (directory / "word_list.json").resolve()
    d_l._create_word_list(path)
    with path.open() as p:
        wl1 = json.load(p)
    tokenizer_list = list(d_l.tokenizer.index_word.values())[0:len(wl1) - 1]
    reserve_token = d_l.tokenizer.index_word[0]

    if os.path.exists(d_l.WORDS_PATH):
        with open(d_l.WORDS_PATH) as p:
            wl2 = json.load(p)
        # ensure files written the same
        # ensure words from tokenizer mirror file -- pad at ind 0 not counted
        assert wl1 == wl2
        assert tokenizer_list == wl1[1:]
        assert reserve_token == wl1[0]

    else:
        assert tokenizer_list == wl1[1:]
        assert reserve_token == wl1[0]


def test_dataset_sizes(d_l, train_batches, val_batches, test_batches):
    _TRAIN_SIZE = tf.data.experimental.cardinality(train_batches).numpy()
    _VAL_SIZE = tf.data.experimental.cardinality(val_batches).numpy()
    _TEST_SIZE = tf.data.experimental.cardinality(test_batches).numpy()
    tolerance = 0.001
    assert (_TRAIN_SIZE * d_l.batch_size /
            d_l.DATASET_SIZE) == pytest.approx(d_l.TRAIN_P, abs=tolerance)
    assert (_VAL_SIZE * d_l.batch_size /
            d_l.DATASET_SIZE) == pytest.approx(d_l.VAL_P, abs=tolerance)
    assert (_TEST_SIZE * d_l.batch_size /
            d_l.DATASET_SIZE) == pytest.approx(d_l.TEST_P, abs=tolerance)


def test_preprocess(d_l, test_batches):
    for i, _ in test_batches:
        break
    i = i[0]
    i_pre = d_l.preprocess_input(i)
    i_unp = d_l.unprocess_input(i_pre)
    assert np.all(np.isclose(i.numpy(), i_unp.numpy(), rtol=0.001))
    assert i.shape == i_pre.shape == i_unp.shape
    assert i.shape == (d_l.IMAGE_SIZE, d_l.IMAGE_SIZE, 3)


def test_create_word_embedding_matrix(d_l):
    matrix = d_l.create_word_embedding_matrix()
    assert matrix.shape == (d_l.vocab_size + 1, d_l.WORD_EMBED_SIZE)


def test_batch_convert_to_words(d_l, test_batches):
    for _, s in test_batches:
        break
    refs = d_l.batch_convert_to_words(s, references=True)
    hyp = tf.convert_to_tensor([l[0] for l in s])
    hyps = d_l.batch_convert_to_words(hyp, references=False)
    assert [l[0] for l in refs] == hyps


"""
models
"""


@pytest.fixture
def encoder():
    encoder = create_encoder(tf.keras.applications.xception.Xception)
    return encoder


@pytest.fixture
def word_embeddings(d_l):
    return d_l.create_word_embedding_matrix()


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("units", [1, 64)
@pytest.mark.parametrize("num_pixels", [1, 100])
def test_attention(batch_size, units, num_pixels):
    dec_state_shape = (units)
    embedder_shape = (num_pixels, units)
    test_attention = create_attention(dec_state_shape, embedder_shape, units)
    embedder_rand = tf.random.normal((batch_size, num_pixels, units))
    dec_rand = tf.random.normal((batch_size, units))
    cvec, attw = test_attention.predict([dec_rand, embedder_rand])
    assert cvec.shape == attw.shape == (batch_size, num_pixels)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("image_size",
                         ['default',
                          pytest.param(100, marks=pytest.mark.xfail()),
                          pytest.param(600, marks=pytest.mark.xfail())])
def test_encoder_no_resize(batch_size, image_size, encoder, d_l):
    if image_size == 'default':
        image_size = d_l.IMAGE_SIZE
    rand_image = tf.random.normal(
        (batch_size, image_size, image_size, 3))

    assert encoder(rand_image).shape == (batch_size, 10, 10, 2048)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("image_size", ["default", 100, 350])
def test_encoder_resize(batch_size, image_size, encoder, d_l):
    default = d_l.IMAGE_SIZE
    if image_size == 'default':
        image_size = default
    rand_image = tf.random.normal(
        (batch_size, image_size, image_size, 3))
    prep_image = d_l.preprocess_input(
        tf.image.resize(rand_image, (default, default)))
    assert encoder(prep_image).shape == (batch_size, 10, 10, 2048)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("embed_dim", [16, 64])
def test_embedder(batch_size, embed_dim):
    enc_out_shape = (batch_size, 10, 10, 2048)
    emb_out = (batch_size, enc_out_shape[1] * enc_out_shape[2], embed_dim)
    enc_out_rand = tf.random.normal(enc_out_shape)
    emb = create_embedder(enc_out_shape, embed_dim)
    assert emb(enc_out_rand).shape == emb_out


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("cvec_shape", [100])
@pytest.mark.parametrize("vocab_size", ["default"])
@pytest.mark.parametrize("units", [32, 64)
@pytest.mark.parametrize("with_embeddings", [True, False])
def test_decoder(batch_size, d_l, cvec_shape, vocab_size, units, with_embeddings, word_embeddings):
    if vocab_size == "default":
        vocab_size = d_l.vocab_size
    if with_embeddings:
        dec = create_decoder((cvec_shape), vocab_size + 1, d_l.WORD_EMBED_SIZE,
                             units, pretrained_embeddings=word_embeddings)
    else:
        dec = create_decoder((cvec_shape), vocab_size + 1,
                             d_l.WORD_EMBED_SIZE, units)
    decoder_input = tf.random.normal((batch_size, cvec_shape))
    initial_word = tf.expand_dims(
        [d_l.tokenizer.word_index['<start>']] * batch_size, 1)
    output, state = dec([initial_word, decoder_input])
    assert output.shape == (batch_size, vocab_size + 1)
    assert state.shape == (batch_size, units)


"""
model_trainer
"""


@pytest.fixture(scope='module')
def m_t():
    return ModelTrainer()


def test_train_step(m_t, train_batches):
    for i, s in train_batches:
        break
    loss = m_t._train_step(i, s, 0)
    assert loss > 0


def test_evaluate(m_t, d_l, val_batches):
    for i, s in val_batches:
        break
    sentences, loss = m_t._evaluate_images(i, s)
    assert loss > 0
    assert sentences.shape == (d_l.batch_size, 19)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_scc_loss(m_t, word_embeddings, batch_size):
    cvec_shape = 100
    dec = create_decoder((cvec_shape), m_t.data_loader.vocab_size + 1,
                         m_t.data_loader.WORD_EMBED_SIZE,
                         64, pretrained_embeddings=word_embeddings)
    initial_word = tf.expand_dims(
        [m_t.data_loader.tokenizer.word_index['<start>']] * batch_size, 1)
    masked_word = tf.expand_dims(
        [m_t.data_loader.tokenizer.word_index['<pad>']] * batch_size, 1)
    cvec = tf.random.normal((batch_size, cvec_shape))
    output, state = dec([initial_word, cvec])
    m_output, m_state = dec([masked_word, cvec])
    assert m_t._scc_loss(initial_word, m_output).numpy() != 0
    assert m_t._scc_loss(initial_word, output).numpy() != 0
    assert m_t._scc_loss(masked_word, m_output).numpy() == 0
    assert m_t._scc_loss(masked_word, output).numpy() == 0


@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("num_pixels", [100, 121])
def test_dsar_loss(m_t, batch_size, num_pixels):
    x = np.zeros((batch_size, 18, num_pixels))
    for i in range(18):
        att_rand = tf.random.normal((batch_size, num_pixels), seed=32)
        x[:, i, :] = att_rand
    assert m_t._dsar_loss(x).shape == tuple()


"""
evaluator
"""


def test_evaluator(d_l, val_batches):
    x = list(val_batches.take(2))
    references = d_l.batch_convert_to_words(x[0][1])
    hypotheses = d_l.batch_convert_to_words(
        d_l.batch_sample_target(x[0][1]), references=False)
    hypotheses2 = d_l.batch_convert_to_words(
        d_l.batch_sample_target(x[1][1]), references=False)

    ev = Evaluator()
    ev.update_scores(references, hypotheses)

    ones = ev.get_scores()
    ones.pop('BLEU_1')
    cdr1 = ones.pop('CIDEr')
    ones = list(ones.values())
    ev.reset_scores()
    ev.update_scores(references, hypotheses2)

    zeros = ev.get_scores()
    zeros.pop('BLEU_1')
    rouge = zeros.pop('ROUGE')
    zeros = list(zeros.values())
    ev.update_scores(references, hypotheses)

    halves = ev.get_scores()
    halves.pop('BLEU_1')
    cdrh = halves.pop('CIDEr')
    halves = list(halves.values())

    assert np.all(np.isclose(ones, [1] * len(ones), rtol=0.01))
    assert cdr1 > 2

    assert np.all(np.isclose(zeros, [0] * len(zeros), atol=0.2))
    assert rouge < 0.3

    assert np.all(np.isclose(halves, [0.5] * len(halves), rtol=0.35))
    assert cdrh > 1


"""
model_driver
"""


@pytest.mark.parametrize("beam_width", [6])
@pytest.mark.parametrize("redundancy", [0.01, 0.99])
@pytest.mark.parametrize("candidates", [0, 5])
@pytest.mark.parametrize("as_words", [True, False])
def test_beam_search(m_t, test_batches, beam_width, redundancy, candidates, as_words):
    m_t.load_model()
    tk = m_t.data_loader.tokenizer
    vocab_size = m_t.data_loader.vocab_size
    for i, s in test_batches:
        break
    i = tf.squeeze(i[0])
    result = beam_search(i,
                         m_t.ckpt.encoder,
                         m_t.ckpt.embedder,
                         m_t.ckpt.attention,
                         m_t.ckpt.decoder,
                         beam_width=beam_width,
                         redundancy=redundancy,
                         candidates=candidates,
                         as_words=as_words)
    if as_words:
        if candidates:
            assert np.all([[(tk.word_index[w] <=
                             vocab_size) for w in c] for c in result])
        else:
            assert np.all([(tk.word_index[w] <= vocab_size) for w in result])
    else:
        if candidates:
            assert np.all([[(s <= vocab_size) for s in c] for c in result])
        else:
            assert np.all([(s <= vocab_size) for s in result])
