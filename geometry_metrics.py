from sklearn.linear_model import RidgeClassifier
from itertools import combinations
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
    
    if num_vertex > embedding_space_dimension:
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
    print("#sampling dichotomy:", len(label_sets_1), len(label_sets_2))

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
         top_K = 3, verbose = False):
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


    svc = RidgeClassifier()
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
        topK_score = all_score[np.argpartition(all_score, -top_K)[-top_K:]]
        return np.mean(topK_score)

