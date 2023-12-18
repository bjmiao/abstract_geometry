from geometry_metrics import generate_grid_in_embedding_space, shattering_dimensionality
import numpy as np

import logging

# Configure the logging settings
logging.basicConfig(filename='grid_sd_1215.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

for grid_size in ( (2, 2), (2, 3), (3, 3), (2, 2, 2), (2, 3, 4)):
    grid_size_str = "_".join([str(x) for x in grid_size])
    for distortion in np.arange(0, 5, 0.5):
        for embedding_dimension in range(50, 301, 50):
            for noise_coef in [0, 0.2, 1]:
                grid = generate_grid_in_embedding_space(grid_size,
                            embedding_space_dimension=embedding_dimension,
                            distortion_magnitude=distortion
                )
                sd = shattering_dimensionality(grid, 5000, noise_coef=noise_coef, sample_dichotomy=100)
                logging.info(f"sd,{grid_size_str},{distortion},{embedding_dimension},{noise_coef},{sd}")
