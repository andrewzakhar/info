import time
import matplotlib.pyplot as plt
import numpy as np
from src import MinkowskiModel, FractalModel


IMAGES_FOLDER = 'images'
SAMPLE_SIZES = [i for i in range(3125, 20000, 1000)]
DIMENSIONS = [i for i in range(5, 10, 1)]

np.random.seed(11)


def get_data_by_dimensions(dimensions, sample_size=3125):
    psi = np.random.uniform(0, np.pi / 2, size=sample_size)
    phi = np.random.uniform(0, np.pi / 2, size=sample_size)
    hi = np.random.uniform(0, np.pi / 2, size=sample_size)
    t = np.random.uniform(0, np.pi / 2, size=sample_size)
    for i in range(len(dimensions)):
        x = np.zeros((sample_size, dimensions[i]))
        x[:, 0] = np.cos(psi) * np.cos(phi) * np.cos(hi) * np.cos(t)
        x[:, 1] = np.cos(psi) * np.sin(phi) * np.cos(hi) * np.cos(t)
        x[:, 2] = np.sin(psi) * np.cos(hi) * np.cos(t)
        x[:, 3] = np.sin(hi) * np.cos(t)
        x[:, 4] = np.sin(t)
        for j in range(5, i):
            x[:, j - 1] = np.cos(np.linspace(0, 1, 3125))
        yield x


def get_data_by_sample_size(sizes, dimension=5):
    for size in sizes:
        psi = np.random.uniform(0, np.pi/2, size=size)
        phi = np.random.uniform(0, np.pi/2, size=size)
        hi = np.random.uniform(0, np.pi/2, size=size)
        t = np.random.uniform(0, np.pi/2, size=size)
        x = np.zeros((size, dimension))
        x[:, 0] = np.cos(psi)*np.cos(phi)*np.cos(hi)*np.cos(t)
        x[:, 1] = np.cos(psi)*np.sin(phi)*np.cos(hi)*np.cos(t)
        x[:, 2] = np.sin(psi)*np.cos(hi)*np.cos(t)
        x[:, 3] = np.sin(hi)*np.cos(t)
        x[:, 4] = np.sin(t)
        yield x


def test_minkowski_dimensions():
    model = MinkowskiModel(verbose=False)
    execution_times_by_dimensions = []
    for sample in get_data_by_dimensions(DIMENSIONS):
        start_time = time.time()
        print(f'compute {sample.shape} dimensions')
        _, n_dims = model.fit_predict(sample)
        if n_dims != 4:
            print('Wrong result')
        print(f'\t number of dimensions = {n_dims}')
        execution_times_by_dimensions.append(time.time() - start_time)
    plt.plot(DIMENSIONS, [i / 60 for i in execution_times_by_dimensions])
    plt.xlabel('dimension size')
    plt.ylabel('time in minutes')
    plt.title('time complexity depending on number of dimensions')
    plt.grid()
    plt.savefig(f'{IMAGES_FOLDER}\\time_minkowski_dimensions.png', quality=100, dpi=400)
    plt.close()


def test_minkowski_sample_size():
    model = MinkowskiModel(verbose=False)
    execution_times_by_sample_size = []
    for item in get_data_by_sample_size(SAMPLE_SIZES):
        start_time = time.time()
        print(f'compute {item.shape} dimensions')
        print(f'number of dimensions = {model.fit_predict(item)}')
        execution_times_by_sample_size.append(time.time() - start_time)
    plt.plot(SAMPLE_SIZES, [i / 60 for i in execution_times_by_sample_size])
    plt.xlabel('sample size')
    plt.ylabel('time in minutes')
    plt.title('time complexity depending on sample size')
    plt.grid()
    plt.savefig(f'{IMAGES_FOLDER}\\time_minkowski_sample_size.png', quality=100, dpi=400)
    plt.close()


def test_fractal_dimensions():
    model = FractalModel(verbose=False)
    execution_times_by_dimensions = []
    for sample in get_data_by_dimensions(DIMENSIONS):
        start_time = time.time()
        print(f'compute {sample.shape} dimensions')
        _, n_dims = model.fit_predict(sample)
        if n_dims != 4:
            print('wrong answer')
        print(f'\t number of dimensions = {n_dims}')
        execution_times_by_dimensions.append(time.time() - start_time)
    plt.plot(DIMENSIONS, [i / 60 for i in execution_times_by_dimensions])
    plt.xlabel('dimension size')
    plt.ylabel('time in minutes')
    plt.title('time complexity depending on number of dimensions')
    plt.grid()
    plt.savefig(f'{IMAGES_FOLDER}\\time_fractal_dimensions.png', quality=100, dpi=400)
    plt.close()


def test_fractal_sample_size():
    model = FractalModel(verbose=False)
    execution_times_by_sample_size = []
    for item in get_data_by_sample_size(SAMPLE_SIZES):
        start_time = time.time()
        print(f'compute {item.shape} dimensions')
        print(f'number of dimensions = {model.fit_predict(item)}')
        execution_times_by_sample_size.append(time.time() - start_time)
    plt.plot(SAMPLE_SIZES, [i / 60 for i in execution_times_by_sample_size])
    plt.xlabel('sample size')
    plt.ylabel('time in minutes')
    plt.title('time complexity depending on sample size')
    plt.grid()
    plt.savefig(f'{IMAGES_FOLDER}\\time_fractal_sample_size.png', quality=100, dpi=400)
    plt.close()


# test_minkowski_dimensions()
# test_minkowski_sample_size()
test_fractal_dimensions()
test_fractal_sample_size()
