import numpy as np
import logging

# Configure the logging settings
logging.basicConfig(filename='../cont_linear_ccgp_2.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

from sklearn.linear_model import LinearRegression
from geometry_metrics import sample_unit_gaussian

lr = LinearRegression()


num_points_training = 1000
num_points_testing = 100

for dim in range(40, 400, 40):
    for num_variable in [2, 3, 4, 5]:
        for noise_coef in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            for mode in ['normal', 'generalization']:
                point_training = np.zeros((num_points_training, dim), dtype='float')
                point_training[:, :num_variable] = np.random.rand(num_points_training, num_variable)

                point_testing = np.zeros((num_points_testing, dim), dtype='float')
                point_testing[:, :num_variable] = np.random.rand(num_points_testing, num_variable)
                if mode == 'generalization':
                    point_testing += 1

                linear_function = lambda cos: sum([i*cos[i] for i in range(num_variable)])

                value_training = np.apply_along_axis(linear_function, axis=1, arr = point_training)
                value_testing = np.apply_along_axis(linear_function, axis=1, arr = point_testing)
                # print(value_training.shape, value_testing.shape)

                point_training_noise = point_training + sample_unit_gaussian(dim, size = (num_points_training, )) * noise_coef
                point_testing_noise = point_testing + sample_unit_gaussian(dim, size = (num_points_testing, )) * noise_coef
                point_training_noise.shape, point_testing_noise.shape
                lr.fit(point_training_noise, value_training)
                value_pred = lr.predict(point_testing_noise)
                score_corr = np.corrcoef(value_pred, value_testing)[0, 1]


                score = lr.score(point_testing_noise, value_testing)
                print(dim, num_variable, noise_coef, mode, score, score_corr)
                logging.info(f"ccgp,{dim},{num_variable},{noise_coef},{mode},{score},{score_corr}")
