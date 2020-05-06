import json
from heapq import heappop, heappush

import numpy as numpy
import tensorflow as tf

from .data_loader import DataLoader as dl


def get_prediction(image_path, checkpoint, candidates=0):
    """
    Gets captions from model based on image

    Parameters
    ----------
    image_path : path to image
    model_trainer : ModelTrainer object from model_trainer module
    candidates : int
        number of captions to return
    """
    image = dl.preprocess_input(dl.load(image_path, path=True))
    captions = beam_search(image,
                           checkpoint.encoder,
                           checkpoint.embedder,
                           checkpoint.attention,
                           checkpoint.decoder,
                           candidates=candidates)
    if candidates == 0:
        captions = [f"{' '.join(captions)}."]
    else:
        captions = [f"{' '.join(caption)}." for caption in captions]
    return captions


def beam_search(image, encoder, embedder, attention, decoder,
                beam_width=30, max_length=18, redundancy=0.4,
                ideal_length=7, candidates=0, as_words=True):
    """
    beam_search is a breadth limited sorted-search: from root <start> take next
    best beam-width children out of vocab_size sorted by probability, then
    calculate children for each child and take next best beam-width children
    out of vocab_size * beamwidth possible children, repeating this process
    until you hit beam-width leaves <end> or a maximum path-size of max_length.
    Each candidate path from root to leaf (or forced end node) is rescored
    based on ideal_length -- different from traditional method of normalizing
    based on caption length and some small value alpha.

    Parameters
    ----------
    image : preprocessed image tensor
        image to be captioned
    encoder : encoder model
    embedder : embedder model
    attention : attention model
    decoder : decoder model
    beam_width : int, greater than 1
        size of scope to find best candidate
    max_length : int
        max tolerated length of caption with <start> and <end> tokens
    redundancy : float from 0 to 1
        percentage of repeated words in caption, high redundancy is nonsensical
    ideal_length : int from 0 to max_length
        represents ideal length of a caption and is used to rescore captions
        to bias those whose length are closest
    candidates : int from 0 to beam_width
        represents number of additional captions to predict
    as_words : bool
        whether output should be a words or encoded as number sequences based
        on word_list.json
    """
    assert beam_width > candidates
    with open(dl.WORDS_PATH, 'r') as fp:
        word_list = json.load(fp)
    bad_endings = ['a', 'with', 'of', 'in', 'on', 'for', 'by']
    bad_endings = [word_list.index(word) for word in bad_endings]

    features = embedder(encoder(tf.expand_dims(image, 0)))
    # to use after root <start>
    features_dup = tf.repeat(features, beam_width, 0)
    decoder_state = tf.zeros((1, decoder.output[1].shape[1]))
    end = word_list.index('<end>')
    word = tf.expand_dims([word_list.index('<start>')], 1)

    # initialize
    context_vector, _ = attention([decoder_state, features])
    word_logits, decoder_state = decoder([word, context_vector])
    scores = tf.nn.log_softmax(word_logits)
    topk = tf.math.top_k(scores, beam_width)
    # minheaps to store scores with tuples (score, sequence tuple, decoder_state)
    # throws ValueError when value of first elem equal another value on heap
    # to resolve make second value guaranteed unique, which the sequence tuple is,
    # also, it will preference more common (smaller indexed) words, a good behavior

    # need to invert values to use minheap as maxheap
    min_heap = []
    candidate_nodes = []

    for i in range(beam_width):
        node = tuple(
            [float(-1 * topk.values[0][i].numpy()),  # word score
             tuple([topk.indices[0][i].numpy()]),  # word
             decoder_state]
        )

        heappush(min_heap, node)

    while len(candidate_nodes) < beam_width:
        nodes = [heappop(min_heap) for i in range(beam_width)]
        min_heap.clear()

        word = tf.reshape([[node[1][-1] for node in nodes]], [beam_width, 1])
        _decoder_state = tf.squeeze(
            tf.concat([[node[2] for node in nodes]], axis=0))

        # get next states and words
        context_vector, _ = attention([_decoder_state, features_dup])
        word_logits, decoder_state = decoder([word, context_vector])

        # get top beamwidth possibilities from each node in nodes
        scores = tf.nn.log_softmax(word_logits)

        topk = tf.math.top_k(scores, beam_width)

        for n, node in enumerate(nodes):

            if len(candidate_nodes) == beam_width:
                break

            # add best nodes to candidates
            # only the nodes that come off the heap should be added
            if node[1][-1] == end:
                if node[1][-2] in bad_endings:
                    continue
                candidate_nodes.append(node)
                continue

            elif len(node[1]) == max_length - 1:
                lst = list(node)
                lst[1] += tuple([end])
                node = tuple(lst)
                candidate_nodes.append(node)
                continue

            # create next nodes to add to heap
            for i in range(beam_width):

                new_word = topk.indices[n][i].numpy()
                new_node = tuple(
                    [(-1 * topk.values[n][i].numpy()) + node[0],
                     node[1] + tuple([new_word]),
                     decoder_state[n]]
                )

                # don't follow nodes that lead to high redundancy
                counts = {}
                for word in new_node[1]:
                    if word not in counts:
                        counts[word] = 0
                    counts[word] += 1
                score = sum(map(lambda x: x if x > 1 else 0,
                                list(counts.values()))) / len(node[1])
                if score >= redundancy:
                    continue

                heappush(min_heap, new_node)

    # collect answer here
    sentences = []

    min_heap.clear()
    # calculate length normalize with alpha and find sentence with best score
    for node in candidate_nodes:
        # alternate ways to rescore captions
        #normalizer = 1 / ((len(node[1]) ** alpha))
        #normalizer = 1 / (abs(ideal_length - len(node[1])) ** alpha + 1)
        score = len(node[1])
        score = (2.5 * ideal_length - score) / 2
        score += node[0]
        new_node = (-score, node[1][:-1])
        heappush(min_heap, new_node)

    if not candidates:
        sentences = heappop(min_heap)[1]

        if as_words:
            sentences = [word_list[i] for i in sentences]

    else:
        sentences = [heappop(min_heap)[1] for n in range(candidates + 1)]
        if as_words:
            sentences = [[word_list[i] for i in n] for n in sentences]

    return sentences
