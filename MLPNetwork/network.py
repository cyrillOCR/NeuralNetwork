import json
import keras
import collections
import numpy as np

from keras.layers import InputLayer, Dense, Dropout


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


f = open("data.json", "r")
a = json.load(f)
features = []
for i in range(0, 471):
    nested_lists = a.get(str(i))
    flat_list = flatten(nested_lists)
    features.append(flat_list)
f.close()

f = open("labels.txt", "r", encoding="utf-8")
labels = []
for line in f:
    line = line.replace("\n", "")
    labels.append(line)

labels_set = set(labels)
labels_mapping = dict()
labels_mapping_reverse = dict()
for index, label in enumerate(labels_set):
    labels_mapping[label] = index
    labels_mapping_reverse[index] = label

features = features[:446]
labels = [labels_mapping[label] for label in labels]

features = np.array(features)
features = features / 128
x_train = features[50:]
y_train = keras.utils.to_categorical(labels[50:], num_classes=33)
x_test = features[:50]
y_test = keras.utils.to_categorical(labels[:50], num_classes=33)

model = keras.models.Sequential()
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
model.add(InputLayer(input_shape=(5824,)))
model.add(Dense(2000, input_shape=(5824,), activation="relu"))
model.add(Dense(33, input_shape=(5824,), activation="softmax"))

model.compile(optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=200, batch_size=5)
model.save('my_model.h5')
