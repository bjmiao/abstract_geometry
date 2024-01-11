from geometry_metrics import generate_hypercube_in_embedding_space, CCGP,\
        shattering_dimensionality, predefine_dimension_for_ccgp
import sys
import pandas as pd
import random
import string

if __name__ == "__main__":
    cube_dimension = 4
    embedding_dimension = int(sys.argv[1])
    noise_coef = float(sys.argv[2])
    distortion_magnitude = float(sys.argv[3])

    a1, a2 = predefine_dimension_for_ccgp(cube_dimension)
    # print(a1, a2)
    # a1 = [[0,1,2,3,4,5,6,7], [0,1,4,5,8,9,12,13], [0,2,4,6,8,10,12,14], [0,1,2,3,8,9,10,11]]
    # a2 = [[x for x in range(16) if x not in a1_this] for a1_this in a1]
    # for embedding_dimension in [10, 30, 50, 80, 100, 200, 300, 400, 500]:
    nsamples = 1000
    nsample_dichotomy = 100
    cube = generate_hypercube_in_embedding_space(cube_dimension, embedding_dimension, distortion_magnitude=distortion_magnitude)
    sd = shattering_dimensionality(
        cube,
        nsamples = nsamples, noise_coef = noise_coef,
        sample_dichotomy = nsample_dichotomy
    )
    ccgp = CCGP(
        cube,
        nsamples = nsamples,
        noise_coef = noise_coef,
        predefined_dichotomy=(a1, a2)
    )
    print("cube_sd_head", cube_dimension, embedding_dimension, noise_coef, distortion_magnitude, sd, ccgp, flush=True)
    #logging.info(f"{cube_dimension},{embedding_dimension},{distortion_magnitude},{noise_coef},{nsamples},{sd},{ccgp}")
