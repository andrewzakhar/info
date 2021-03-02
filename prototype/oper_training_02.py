import keras
from keras import layers
from keras import Input
from keras.models import Model

def net_with_three_inputs(train_x, train_y, test_x, test_y,
                          opers_voc, params_voc, params_values_voc,
                          oper_input, param_input, param_value_input,
                          oper_test, param_test, param_value_test):

    oper_in = keras.Input(shape=(None,), name='oper_input')
    param_in = keras.Input(shape=(None,), name='param_input')
    param_value_in = keras.Input(shape=(None, 10), name='param_value_input')

    embedded_oper = layers.Embedding(opers_voc, 64)(oper_in)
    embedded_param = layers.Embedding(params_voc, 64)(param_in)

    concatenated = layers.concatenate([embedded_oper, embedded_param, param_value_in])

    x_lstm1 = layers.LSTM(32, return_sequences=True)(concatenated)
    x_lstm2 = layers.LSTM(32, return_sequences=False)(x_lstm1)

    x_dense = layers.Dense(10, activation="relu")(x_lstm2)

    # Добавление классификатора softmax сверху
    predict = layers.Dense(opers_voc, activation='softmax')(x_dense)

    # Создание экземпляра модели с тремя входами и одним выходом
    model = Model([oper_in, param_in, param_value_in], predict)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    # Отображение слоев
    keras.utils.plot_model(model, 'multi_input.png', show_shapes=True)

    # Передача списка входов
    model.fit([oper_input, param_input, param_value_input], train_y,
              epochs=10)
              # validation_data=([oper_test, param_test, param_value_test], test_y))

    return model