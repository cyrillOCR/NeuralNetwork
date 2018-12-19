import json
import collections
import numpy as np
import keras
from keras.models import load_model


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

features = features[0:446]
labels = [labels_mapping[label] for label in labels]

features = np.array(features)
features = features / 128

y = keras.utils.to_categorical(labels[:50], num_classes=33)

model = load_model("my_model.h5")
loss, accuracy = model.evaluate(features[0:50],y[0:50])
print("Loss: {}; Acc: {}".format(loss, accuracy))
predictions = model.predict(features[0:50])
predictions = predictions.argmax(axis=-1)
predictions = np.ndarray.tolist(predictions)
predictions = [labels_mapping_reverse[index] for index in predictions]
print(predictions)
prediction = "".join(predictions)

actual = [labels_mapping_reverse[index] for index in labels[0:50]]
actual_str = "".join(actual)
print("Actual string:")
print(actual_str)
print("Predicted string:")
print(prediction)
