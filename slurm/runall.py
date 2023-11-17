import os
import subprocess
import numpy as np

filename = "batch.sbatch" 

for embedding_dimension in range(40, 400, 20):
    for noise_coef in [0.2, 1]:
        for distortion_magnitude in np.arange(0, 4, 0.25):
            p = subprocess.Popen(["sbatch", filename, str(embedding_dimension), str(noise_coef), str(distortion_magnitude)], stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE)
            out, err = p.communicate()
            print(out.decode(), err.decode())
