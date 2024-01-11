import os
import subprocess
import numpy as np

filename = "batch.sbatch" 
executive_name = "../calc_cube_sd_ccgp.py"

for embedding_dimension in range(40, 401, 20):
    for noise_coef in [0, 0.2, 1]:
        for distortion_magnitude in np.arange(0, 4.01, 0.25):
            p = subprocess.Popen(["sbatch", filename, executive_name, str(embedding_dimension), str(noise_coef), str(distortion_magnitude)], stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE)
            out, err = p.communicate()
            print(out.decode(), err.decode())
