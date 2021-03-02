
import numpy as np
import tensorflow as tf

def one_hot_encoding_and_flatten(data2d, n_classes):
    ohe = tf.keras.utils.to_categorical(data2d, n_classes)
    # Вероятно, это объединение можно сделать непосредственно через numpy?
    if (len(ohe.shape) == 2):
        flatten_size = ohe.shape[1]
    else:
        flatten_size = ohe.shape[1] * ohe.shape[2]
    flatten = np.empty((ohe.shape[0], flatten_size), float)
    for i in range(ohe.shape[0] - 1):
        flatten[i] = ohe[i].flatten()
    return flatten

# Генератор OHE-векторов
class OneHotEncodedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, n_classes, batch_size=32):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.on_epoch_end()

        self.next_index = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(int(self.x.shape[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        batch_x = self.x[self.next_index:self.next_index + self.batch_size]
        batch_y = self.y[self.next_index:self.next_index + self.batch_size]

        batch_x_ohe = one_hot_encoding_and_flatten(batch_x, self.n_classes)
        batch_y_ohe = one_hot_encoding_and_flatten(batch_y, self.n_classes)

        self.next_index = self.next_index + self.batch_size

        return batch_x_ohe, batch_y_ohe

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.next_index = 0


class CustomModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_x, test_y):
        self.test_x = test_x
        self.test_y = test_y
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('acc') >= 0.9:
            self.model.stop_training = True
        test_lost, test_acc = self.model.evaluate(self.test_x, self.test_y, verbose=0)
        print('\ntest loss: {}, acc: {}\n'.format(test_lost, test_acc))