from geometry_metrics import generate_grid_in_embedding_space, shattering_dimensionality, CCGP_grid_lastdim
import numpy as np

import logging

# Configure the logging settings
logging.basicConfig(filename='grid_ccgp_1215.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# for grid_size in ( (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)):
for grid_size in ( (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2)):
    grid_size_str = "_".join([str(x) for x in grid_size])
    for distortion in np.arange(0, 5, 1):
        for embedding_dimension in range(50, 201, 50):
            for noise_coef in [0, 0.2, 1]:
                grid = generate_grid_in_embedding_space(grid_size,
                            embedding_space_dimension=embedding_dimension,
                            distortion_magnitude=distortion
                )
                ccgp = CCGP_grid_lastdim(grid, grid_size)
                logging.info(f"ccgp,{grid_size_str},{distortion},{embedding_dimension},{noise_coef},{ccgp}")
