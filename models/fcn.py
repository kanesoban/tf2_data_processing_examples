import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_ORDERING = 'channels_last'


def encoder(input_height, input_width):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    # img_input = layers.Input(shape=(input_height, input_width, 3))
    img_input = layers.Input(shape=(input_height, input_width, 1))
    x = img_input
    levels = []

    x = (layers.ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (layers.Conv2D(filter_size, (kernel, kernel), data_format=IMAGE_ORDERING, padding='valid'))(x)
    x = (layers.BatchNormalization())(x)
    x = (layers.Activation('relu'))(x)
    x = (layers.MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    x = (layers.ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (layers.Conv2D(128, (kernel, kernel), data_format=IMAGE_ORDERING, padding='valid'))(x)
    x = (layers.BatchNormalization())(x)
    x = (layers.Activation('relu'))(x)
    x = (layers.MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    for _ in range(3):
        x = (layers.ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
        x = (layers.Conv2D(256, (kernel, kernel), data_format=IMAGE_ORDERING, padding='valid'))(x)
        x = (layers.BatchNormalization())(x)
        x = (layers.Activation('relu'))(x)
        x = (layers.MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
        levels.append(x)

    return img_input, levels


def crop(output1, output2, i):
    o_shape2 = keras.Model(i, output2).output_shape

    output_height2 = o_shape2[1]
    output_width2 = o_shape2[2]

    o_shape1 = keras.Model(i, output1).output_shape

    output_height1 = o_shape1[1]
    output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        output1 = layers.Cropping2D(cropping=((0, 0), (0, cx)), data_format=IMAGE_ORDERING)(output1)
    else:
        output2 = layers.Cropping2D(cropping=((0, 0), (0, cx)), data_format=IMAGE_ORDERING)(output2)

    if output_height1 > output_height2:
        output1 = layers.Cropping2D(cropping=((0, cy), (0, 0)), data_format=IMAGE_ORDERING)(output1)
    else:
        output2 = layers.Cropping2D(cropping=((0, cy), (0, 0)), data_format=IMAGE_ORDERING)(output2)

    return output1, output2


def get_segmentation_model(input, output) -> keras.Model:
    img_input = input

    o_shape = keras.Model(img_input, output).output_shape
    i_shape = keras.Model(img_input, output).input_shape

    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]
    # Why reshape in original code ?
    #output = (layers.Reshape((output_height * output_width, -1)))(output)

    output = (layers.Activation('softmax'))(output)
    model = keras.Model(img_input, output)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    '''
    model.train = keras.MethodType(train, model)
    model.predict_segmentation = keras.MethodType(predict, model)
    model.predict_multiple = keras.MethodType(predict_multiple, model)
    model.evaluate_segmentation = keras.MethodType(evaluate, model)
    '''

    return model


def fcn_8(n_classes: int, input_height: int, input_width: int) -> keras.Model:
    img_input, levels = encoder(input_height, input_width)

    [f1, f2, f3, f4, f5] = levels

    output1 = f5

    output1 = (layers.Conv2D(4096, (7, 7), activation='relu', padding='same', data_format=IMAGE_ORDERING))(output1)
    output1 = layers.Dropout(0.5)(output1)
    output1 = (layers.Conv2D(4096, (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING))(output1)
    output1 = layers.Dropout(0.5)(output1)
    output1 = (layers.Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(output1)
    output1 = layers.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False,
                                     data_format=IMAGE_ORDERING)(output1)

    output2 = f4

    output2 = (layers.Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(output2)
    output1, output2 = crop(output1, output2, img_input)

    output1 = layers.Add()([output1, output2])

    output1 = layers.Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False,
                                     data_format=IMAGE_ORDERING)(output1)
    output2 = f3

    output2 = (layers.Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(output2)
    output2, output1 = crop(output2, output1, img_input)
    output1 = layers.Add()([output2, output1])
    '''
    output1 = layers.Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False,
                                   data_format=IMAGE_ORDERING)(output1)
    '''
    output1 = layers.Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False,
                                     data_format=IMAGE_ORDERING)(output1)
    model = get_segmentation_model(img_input, output1)
    model.model_name = "fcn_8"

    return model
