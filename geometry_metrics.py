from sklearn.linear_model import RidgeClassifier
from itertools import combinations
from functools import reduce
import random
import numpy as np
import pandas as pd


def sample_distortion(dimension, magnitude, size = 1):
    '''
        return a vector with size (dimension, size), where each row
        represents a random point in the unit ball, then scaled by magnitude 
    '''
    vecs = []
    for _ in range(size):
        vec = np.random.randn(dimension)
        vec = vec / np.linalg.norm(vec) * magnitude
        vecs.append(vec)
    return np.array(vecs)

def sample_unit_gaussian(dim, size = None):
    '''
        return sample N multivariate normal gaussian points
        shape (dimension, size)
    '''
    mean = np.zeros(dim)  # Mean vector of zeros
    cov = np.eye(dim)     # Identity covariance matrix
    sample = np.random.multivariate_normal(mean, cov, size = size)
    return sample

def span_to_gaussian_cloud(points, nsamples, noise_coef):
    '''
        span a set of points into sets of gaussian clond
        points: (num_points, shape)
        return: (num_points * nsamples, shape)
    '''
    point_clouds = []
    dim = points.shape[1]

    for i in range(points.shape[0]):
        central = points[i][np.newaxis, :]
        point_clouds.append(central + sample_unit_gaussian(dim = dim, size=nsamples) * noise_coef)
    return np.concatenate(point_clouds, axis = 0)

def generate_hypercube_in_embedding_space(cube_dimension, embedding_space_dimension, distortion_magnitude = 0, choice = [0, 1]):
    '''
        generate a hypercube embedded in a high-dimensional space
        return: (2**cube_dimension, embedding_space_dimension)
    '''
    if len(choice) != 2:
        raise ValueError("Now we only support choice number is 2")
    num_vertex = len(choice) ** cube_dimension
    
    if cube_dimension > embedding_space_dimension:
        raise ValueError("The cube dimension is higher than space dimension")
    hi_dim_cube = np.zeros(
        (num_vertex, embedding_space_dimension),
        dtype = 'float'
    )
    # we are using bit operation to get the dimension
    all_index = list(range(num_vertex))
    for i in range(cube_dimension):
        now_index = np.bitwise_and(all_index, (1 << i))
        hi_dim_cube[now_index == 0, i] = choice[0]
        hi_dim_cube[now_index > 0, i] = choice[1]
    if distortion_magnitude > 0:
        hi_dim_cube = hi_dim_cube + sample_distortion(embedding_space_dimension, distortion_magnitude, size = num_vertex)
    return hi_dim_cube

def generate_grid_in_embedding_space(dim_vertex_numbers, embedding_space_dimension,
        distortion_magnitude = 0, interval = 1,
        show_edges = False):
    '''
        dim_vertex_numbers: The number of vertex on each dimension. E.g. (2, 3) returns a 日 grid
        embedding_space_dimension: same as the hypercube case. Deternimes the vector length of each vertex
        distortion_magnitude: same
        interval: Number or list. The intervel between different vertices along each direction. Number means a uniform interval.
    '''
    num_dim = len(dim_vertex_numbers)
    num_vertex = reduce(lambda a,b: a*b, dim_vertex_numbers)
    print(num_dim, num_vertex)
    if type(interval) is float or type(interval) is int:
        pass
    else:
        raise NotImplementedError("Grid other than uniform interval is not implemented yet")
    hi_dim_grid = np.zeros(
        (num_vertex, embedding_space_dimension),
        dtype = 'float'
    )
    base = 1
    for i in range(num_dim):
        group_repeat = base
        label_repeat = num_vertex // dim_vertex_numbers[i] // group_repeat
        # print(group_repeat, label_repeat)
        now_index = list(np.repeat(range(dim_vertex_numbers[i]), label_repeat)) * group_repeat
        # print(i, len(now_index), now_index)
        hi_dim_grid[:, i] = now_index
        base = base * dim_vertex_numbers[i]
    if show_edges:
        # constructed all the edges
        # TODO: independent of hi_dim_grid
        edges = set()
        for vid in range(num_vertex):
            # a vertex has edge on each direction that it connects to the prev and next node
            adjacent_vid_diff = num_vertex
            for dim in range(num_dim):
                adjacent_vid_diff = adjacent_vid_diff // dim_vertex_numbers[dim]
                if hi_dim_grid[vid, dim] > 0:
                    edges.add((vid-adjacent_vid_diff, vid)) # keep the edge (small, large) so that no repeat occurs
                if hi_dim_grid[vid, dim] < dim_vertex_numbers[dim]-1:
                    edges.add((vid, vid+adjacent_vid_diff))

    if distortion_magnitude > 0:
        hi_dim_grid = hi_dim_grid + sample_distortion(embedding_space_dimension, distortion_magnitude, size = num_vertex)
    if show_edges:
        return hi_dim_grid, edges
    else:
        return hi_dim_grid

def shattering_dimensionality(
        points, nsamples = 100, noise_coef = 0.2,
        sample_dichotomy = None, verbose = False
    ):
    '''
        calculate the shattering dimensionality of a manifold
        Points: num_points, dimensions
        nsamples: #points in the gaussian cloud
        noise_coef: noise coeffient of the gaussian
        sample_dichotomy: None or an integer. Do we sample a subset of the dichotomies?
        verbose: whether we return the score of each dichotomy
    '''
    num_points = points.shape[0]
    all_label = list(range(num_points))
    # get non-repeated dichotomy
    label_sets_1 = [x for x in combinations(all_label, num_points // 2) if 0 in x] # to avoid repetetion
    if sample_dichotomy is not None and sample_dichotomy > 0:
        random.shuffle(label_sets_1)
        label_sets_1 = label_sets_1[:sample_dichotomy]
    label_sets_2 = [tuple([x for x in range(num_points) if not x in label_set1]) for label_set1 in label_sets_1]
    # print("#sampling dichotomy:", len(label_sets_1), len(label_sets_2))

    svc = RidgeClassifier(alpha=0.01)
    # separable_count = 0
    decoding_scores = []
    for label_set1, label_set2 in zip(label_sets_1, label_sets_2):
        point_set_1 = points[label_set1, :]
        point_set_2 = points[label_set2, :]

        # add gaussion noise
        pcloud1 = span_to_gaussian_cloud(point_set_1, nsamples=nsamples, noise_coef=noise_coef)
        pcloud2 = span_to_gaussian_cloud(point_set_2, nsamples=nsamples, noise_coef=noise_coef)
    
        X_train = np.concatenate([pcloud1, pcloud2], axis = 0)
        y_train = np.array([0] * len(pcloud1) + [1] * len(pcloud2))

        pcloud3 = span_to_gaussian_cloud(point_set_1, nsamples=nsamples, noise_coef=noise_coef)
        pcloud4 = span_to_gaussian_cloud(point_set_2, nsamples=nsamples, noise_coef=noise_coef)

        X_test = np.concatenate([pcloud3, pcloud4], axis = 0)
        y_test = np.array([0] * len(pcloud3) + [1] * len(pcloud4))
    
        svc.fit(X_train, y_train)
        score = svc.score(X_test, y_test)
        decoding_scores.append(score)
        # if score > 1 - 1e-4: # if fully decodable
        #     separable_count += 1 
    # return separable_count / len(label_sets_1)
    decoding_scores = np.array(decoding_scores)
    if verbose:
        return decoding_scores
    else:
        return np.mean(decoding_scores) # separable_count / len(label_sets_1)


def CCGP(points, nsamples = 100, noise_coef = 0.2,
         predefined_dichotomy = None, sample_dichotomy = None, sample_training_split = None,
         verbose = False):
    '''
        Calculate the CCGP of a manifold
        Points: (num_points, dimension)
        nsamples, noise_coef: same
        predifined_dichotomy: pass two lists of dichotomies.
        sample_dichotomy: same
        sample_training_split: sample a subset of the training/test split
        verbose: whether we return the score of each dichotomy
    '''
    num_points = points.shape[0]
    all_label = list(range(num_points))
    # get non-repeated dichotomy
    separable_count = 0

    label_sets_1, label_sets_2 = None, None
    if predefined_dichotomy is not None:
        label_sets_1, label_sets_2 = predefined_dichotomy[0], predefined_dichotomy[1]
    else:
        label_sets_1 = [x for x in combinations(all_label, num_points // 2) if 0 in x] # to avoid repetetion
        if sample_dichotomy is not None and sample_dichotomy > 0:
            random.shuffle(label_sets_1)
            label_sets_1 = label_sets_1[:sample_dichotomy]
        label_sets_2 = [tuple([x for x in range(num_points) if not x in label_set1]) for label_set1 in label_sets_1]

    svc = RidgeClassifier(alpha=0.01)
    all_score = []
    training_ratio = 0.8
    for label_set1, label_set2 in zip(label_sets_1, label_sets_2):
        label_sets_training_1 = list(combinations(label_set1, int(len(label_set1) * training_ratio)))
        label_sets_training_2 = list(combinations(label_set2, int(len(label_set2) * training_ratio)))
        if sample_training_split is not None and sample_training_split > 0:
            random.shuffle(label_sets_training_1)
            label_sets_training_1 = label_sets_training_1[:sample_training_split]
            label_sets_training_2 = label_sets_training_2[:sample_training_split]

        dichotomy_score = []
        for train_p1 in label_sets_training_1:
            for train_p2 in label_sets_training_2:
                train_p1 = list(train_p1)
                train_p2 = list(train_p2)
                test_p1 = [x for x in label_set1 if x not in train_p1]
                test_p2 = [x for x in label_set2 if x not in train_p2]
                X_train = np.concatenate(
                    [
                        span_to_gaussian_cloud(points[train_p1, :], nsamples, noise_coef),
                        span_to_gaussian_cloud(points[train_p2, :], nsamples, noise_coef)
                    ], axis = 0)
                y_train = np.array([0] * len(train_p1) * nsamples + [1] * len(train_p2) * nsamples)

                X_test = np.concatenate(
                    [
                        span_to_gaussian_cloud(points[test_p1, :], nsamples, noise_coef),
                        span_to_gaussian_cloud(points[test_p2, :], nsamples, noise_coef)
                    ], axis = 0)
                y_test = np.array([0] * len(test_p1) * nsamples + [1] * len(test_p2) * nsamples)

                svc.fit(X_train, y_train)
                score = svc.score(X_test, y_test)
                # all_score.append(score)
                dichotomy_score.append(score)
        all_score.append(np.mean(dichotomy_score))
    # find the first K max scores
    all_score = np.array(all_score)
    if verbose:
        return all_score
    else:
        top_K = len(label_sets_1)
        topK_score = all_score[np.argpartition(all_score, -top_K)[-top_K:]]
        return np.mean(topK_score)

def predefine_dimension_for_ccgp(cube_dimension):
    '''
        Get pre-defined dimension for maximum ccgp calculation
    '''
    all_vertex = np.arange(2 ** cube_dimension)
    a1 = []
    a2 = []
    for i in range(cube_dimension):
        # get all (id & i == 0) vertex
        a1.append(np.where(np.bitwise_and(all_vertex, 1 << i))[0])
        a2.append(np.where(np.logical_not(np.bitwise_and(all_vertex, 1 << i)))[0])
    return a1, a2


def linear_decoding_score_span_gaussion(points,
        train_points_index, train_labels, test_points_index, test_labels,
        nsamples = 100, noise_coef = 0.0
    ):
    '''
        This is a convenient wrapper for getting the result of a decoder quickly.
    '''

    X_train = span_to_gaussian_cloud(points[train_points_index, :], nsamples, noise_coef)
    y_train = np.repeat(train_labels, nsamples)


    X_test = span_to_gaussian_cloud(points[test_points_index, :], nsamples, noise_coef)
    y_test = np.repeat(test_labels, nsamples)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    svc = RidgeClassifier(alpha=0.01)
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    # all_score.append(score)
    return score

from sklearn.linear_model import LinearRegression
def linear_regression_score_span_gaussion(points,
        train_points_index, train_labels, test_points_index, test_labels,
        nsamples = 100, noise_coef = 0.0
    ):

    X_train = span_to_gaussian_cloud(points[train_points_index, :], nsamples, noise_coef)
    y_train = np.repeat(train_labels, nsamples)


    X_test = span_to_gaussian_cloud(points[test_points_index, :], nsamples, noise_coef)
    y_test = np.repeat(test_labels, nsamples)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    score = np.corrcoef(y_pred, y_test)[0, 1]
    # svc = RidgeClassifier()
    # svc.fit(X_train, y_train)
    # score = svc.score(X_test, y_test)
    # all_score.append(score)
    return score


def CCGP_grid_lastdim(points, grid_shape, nsamples = 100, noise_coef = 0.2):
    '''
        Calculate the CCGP of a grid and take the last dimension as decoding variable
        points:
        grid_shape: used to determine which directions we care about
        decoding_dimensions: None or integer or list
    '''
    num_dim = len(grid_shape)
    num_vertex = points.shape[0]
     
    # we first decode along the last dimension
    num_context = num_vertex // grid_shape[-1]
    num_label = grid_shape[-1]
    # num_training_context = num_context-1
    # num_test_context = 1
    score_all = []
    for test_context in range(num_context):
        test_points_base = num_label * test_context
        test_points_index = list(range(test_points_base, test_points_base + num_label))
        test_label = list(range(num_label))
        train_points_index = [x for x in range(num_vertex) if x not in test_points_index]
        train_label = list(range(num_label)) * (num_context - 1)
        print(train_points_index, train_label, test_points_index, test_label)
        score = linear_regression_score_span_gaussion(points, train_points_index, train_label,
                                            test_points_index, test_label, noise_coef=noise_coef)
        score_all.append(score)
    return np.mean(score_all)
