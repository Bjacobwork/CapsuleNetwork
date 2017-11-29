import numpy as np

def build_skew_matrix(one_hot, skews, index, depth):
    assert len(one_hot) == len(skews), "Length of one_hot must equal skews!"
    depth = depth - 1
    skew_hot = []
    for i, skew in enumerate(skews):
        skew_hot.append(np.multiply(one_hot[i], skew))
    skew_hot = np.expand_dims(skew_hot, axis=2)

    left = []
    right = []
    if index != 0:
        left = np.zeros(shape=[len(one_hot), len(one_hot[0]), index])
    if index != depth:
        right = np.zeros(shape=[len(one_hot), len(one_hot[0]), depth - index])

    if len(left) != 0 and len(right) != 0:
        skew_matrix = np.concatenate([left, skew_hot, right], axis=2)
    elif len(left) != 0:
        skew_matrix = np.concatenate([left, skew_hot], axis=2)
    elif len(right) != 0:
        skew_matrix = np.concatenate([skew_hot, right], axis=2)
    else:
        skew_matrix = skew_hot

    return skew_matrix

def build_mask_matrix(one_hot, indicies, depth):
    vals = np.zeros(shape=[depth])
    vals[indicies] = 1
    vals = np.tile(np.expand_dims(np.expand_dims(vals, axis=0), axis=0), [len(one_hot),len(one_hot[0]),1])
    mask = np.tile(np.expand_dims(one_hot, axis=2), [1,1,depth])
    return np.multiply(mask, vals)

