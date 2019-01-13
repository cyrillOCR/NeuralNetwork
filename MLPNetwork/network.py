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



input_shape = 454
files = ["data3.json","data4.json","data5.json","data6.json","data7.json","data8.json"]
features = []
for file in files:
    f = open(file, "r")
    a = json.load(f)
    for nested_lists in a.values():
        flat_list = flatten(nested_lists)
        features.append(flat_list)
    f.close()

files = ["labels3.txt","labels4.txt","labels5.txt","labels6.txt","labels7.txt","labels8.txt"]
labels = []
for file in files:
    f = open(file, "r", encoding="utf-8")
    for line in f:
        line = line.replace("\n", "")
        labels.append(line)
    f.close()
#
# labels_set = set(labels)
# l_l = list(labels_set)
# l_l.sort()
# f = open("labelsset.txt","w",encoding="utf-8")
# for el in l_l:
#     f.write(el)
#     f.write("\n")
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
model.fit(x_train, y_train, epochs=15, batch_size=5)
keras.models.save_model(model, "saved_model.h5", overwrite=True, include_optimizer=True)
