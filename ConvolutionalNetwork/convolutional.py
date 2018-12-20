from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
from keras.optimizers import Adadelta, RMSprop, SGD
import numpy as np

# dimensions of our images.
img_width, img_height = (28, 28)
# set parameters
N_LABELED = 256
N_UNLABELED = 4 * 1024
EPOCHS = 10
BATCH_SIZE = 8
# make program compatible with both theano and tf
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
# load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshape and normalize
x_train = x_train.reshape((60000,) + input_shape)
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((10000,) + input_shape)
x_test = x_test.astype('float32') / 255
# create one-hot vectors from labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# select N_LABELED labeled training examples
x_train_labeled = x_train[:N_LABELED]
y_train = y_train[:N_LABELED]
# select N_UNLABELED unlabeled training examples
x_train_unlabeled = x_train[N_LABELED:N_LABELED+N_UNLABELED]
x_test_unlabeled = x_test[N_LABELED:N_LABELED+N_UNLABELED]
x_test = x_test[N_LABELED+N_UNLABELED:]
y_test = y_test[N_LABELED+N_UNLABELED:]
# create neural net model
input_img = Input(shape=input_shape)
model = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
model = MaxPooling2D((2, 2), padding='same')(model)
model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)
model = MaxPooling2D((2, 2), padding='same')(model)
model = Flatten()(model)
model = Dense(10, activation='softmax')(model)
model = Model(input_img, model)
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['acc'])
# save weights so that we can use them later in the experiment
model.save_weights('weights.h5')
avg_acc = 0.0
for i in range(30):
    # train model
    model.fit(x_train_labeled, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=0)
    # evaluate model and get accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    avg_acc += test_acc
    # reload weights so that we can retrain the model in 
    # the next iteration
    model.load_weights('weights.h5')
print('Average accuracy:', avg_acc / 30)