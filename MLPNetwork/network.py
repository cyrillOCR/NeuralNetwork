import json
import keras
import collections
import numpy as np
from keras.layers import InputLayer, Dense


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]



input_shape = 706
file = "training_features.json"
features = []
f = open(file, "r")
a = json.load(f)
for nested_lists in a.values():
    flat_list = flatten(nested_lists)
    features.append(flat_list)
f.close()



f = open("training_labels.txt", "r", encoding="utf-8")
labels = []
for line in f:
    line = line.replace("\n", "")
    labels.append(line)


f = open("labelsset.txt", "r", encoding="utf-8")
labels_set = []
for line in f:
    line = line.replace("\n", "")
    labels_set.append(line)
f.close()
classes = len(labels_set)


labels_mapping = dict()
labels_mapping_reverse = dict()
for index, label in enumerate(labels_set):
    labels_mapping[label] = index
    labels_mapping_reverse[index] = label


labels = [labels_mapping[label] for label in labels]
features = np.array(features)
features = features / 128
x_train = features
y_train = keras.utils.to_categorical(labels, num_classes=classes)

model = keras.models.Sequential()
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
model.add(InputLayer(input_shape=(input_shape,)))
model.add(Dense(500, input_shape=(input_shape,), activation="relu"))
model.add(Dense(classes, input_shape=(input_shape,), activation="softmax"))

model.compile(optimizer=keras.optimizers.Adam(),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=5)
keras.models.save_model(model, "saved_model.h5", overwrite=True, include_optimizer=True)
