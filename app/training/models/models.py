import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (GRU, Add, Concatenate, Dense, Embedding,
                                     Input, Lambda, Multiply, ReLU, Reshape,
                                     Softmax)
from tensorflow.keras.models import Model


def create_attention(dec_state_shape, embedder_out_shape, att_units):
    """
    Creates attention soft-attention model

    Parameters
    ----------
    dec_state_shape : 1-tuple of int
        shape of decoder hidden state
    embedder_out_shape : 2-tuple of int
        shape of embedder output
    att_units : int
        size of 2 dense layers connecting encoder and decoder inputs
    """
    enc_out = Input(embedder_out_shape)  # (batch, 100, embed)
    _dec_state = Input(dec_state_shape)  # (batch, h_units)
    dec_state = Lambda(lambda x: K.expand_dims(x, 1))(
        _dec_state)  # (batch, 1, h_units)

    enc_w = Dense(att_units, name="encoder_weights")(
        enc_out)  # (batch, 100, units)
    dec_w = Dense(att_units, name="decoder_weights")(
        dec_state)  # (batch, 1 , units)

    # broadcasts : (batch, 100, units)
    _score = Add(name="add")([enc_w, dec_w])
    score = Dense(1, name="dense_score")(_score)  # (batch, 100, 1)
    sq_score = Lambda(lambda x: K.squeeze(x, axis=-1),
                      name="squeeze_score")(score)  # (batch, 100)

    _att_w = Softmax(name="softmax")(sq_score)  # (batch, 100)
    att_w = Lambda(lambda x: K.expand_dims(x, 2),
                   name='expand_dims')(_att_w)  # (batch, 100, 1)

    _context_vec = Multiply(name="multiply_cvec_enc")(
        [att_w, enc_out])  # (batch, 100, embed)
    context_vec = Lambda(lambda x: K.sum(x, axis=-1),
                         name="sum_cvec")(_context_vec)  # (batch, 100)

    return Model([_dec_state, enc_out], [context_vec, _att_w])


def create_encoder(base_model):
    """
    Creates encoder based on base_model

    Parameters
    ----------
    base_model : model in keras.applications
        used to create base model without classification top and trained on
        imagenet
    """
    return base_model(weights='imagenet', include_top=False)


def create_embedder(enc_out_shape, embed_dim):
    """
    Creates embedder model over encoder

    Parameters
    ----------
    enc_out_shape : 3-tuple of int
        output shape of encoder
    embed_dim : int
        size of dense layer to reshape last dimension of encoder output
    """
    enc_output = Input(enc_out_shape)
    x = Reshape((-1, enc_out_shape[-1]))(enc_output)  # (batch, 100, 4032)
    x = Dense(embed_dim)(x)  # (batch, 100, embed_dim)
    x = ReLU()(x)

    return Model(enc_output, x)


def create_decoder(context_vec_shape, vocab_size, embed_size,
                   dec_units, pretrained_embeddings=None):
    """
    Creates GRU decoder with soft-attention and optional pretrained embeddings

    Parameters
    ----------
    context_vec_shape : 1-tuple of int
        shape of context vector output from attention model
    vocab_size : int
        used to produce tensor of vocab_size logits to predict word
    embed_size : int
        size of embedding layer, if using glove then it's 300
    dec_units : int
        size of GRU layer
    pretrained_embeddings : array of shape vocab_size x embed_size (default None)
        can use create_word_embedding_matrix in data_loader for glove embeddings
    """
    # (batch, 1)
    word = Input((1), name="word_input")

    # (batch, 100)
    context_vec = Input(context_vec_shape, name="context_vector_input")

    # if mask_zero=True, how to propogate to rest of model?
    # label softening? initializer?
    # (batch, 1, embed_dim)
    if pretrained_embeddings is not None:
        embed_word = Embedding(vocab_size, embed_size, weights=[
            pretrained_embeddings], name="word_embedding",
            trainable=False)(word)
    else:
        embed_word = Embedding(vocab_size, embed_size, name="word_embedding",
                               trainable=True)(word)

    # (batch, 1, 100)
    e_context_vec = Lambda(lambda x: K.expand_dims(
        x, 1), name="expand_dims")(context_vec)

    # (batch, 1, 100 + embed_dim)
    concat = Concatenate(name="concatenate")([e_context_vec, embed_word])

    output, state = GRU(dec_units, return_state=True,
                        recurrent_initializer='glorot_uniform', name="GRU")(concat)
    output = Dense(dec_units, name="units_dense")(output)
    output = Dense(vocab_size, name="vocab_dense")(output)

    return Model([word, context_vec], [output, state])
