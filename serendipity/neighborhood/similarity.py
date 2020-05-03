from sklearn.metrics.pairwise import pairwise_distances


def similarity_matrix(mat):
    # sim = - pairwise_distances(mat, metric='euclidean')
    # return pp.minmax_scale(sim)
    return 1 - pairwise_distances(mat, metric='cosine')
