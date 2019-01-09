import keras
import pickle
import gzip
from keras.layers import InputLayer, Dense, Dropout

def load_dataset():
    with gzip.open("mnist.pkl.gz", "rb") as f:
        return pickle.load(f, encoding="latin1")


train_set, valid_set, test_set = load_dataset()

x_train = train_set[0]
# facem vectorul unde avem 1 pe pozitia targetului de la 0-10
y_train = keras.utils.to_categorical(train_set[1], num_classes=10)
x_test = test_set[0]
y_test = keras.utils.to_categorical(test_set[1], num_classes=10)
x_valid = valid_set[0]
y_valid = keras.utils.to_categorical(valid_set[1], num_classes=10)

model = keras.models.Sequential()
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
model.add(InputLayer((784,)))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss="cosine_proximity", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=30, )

loss, accuracy = model.evaluate(x_test, y_test)
print("Loss: {}; Acc: {}".format(loss, accuracy))
