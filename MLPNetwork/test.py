import json
import collections
import numpy as np
import keras


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


file = "test_features.json"
features = []
f = open(file, "r")
a = json.load(f)
for nested_lists in a.values():
    flat_list = flatten(nested_lists)
    features.append(flat_list)
f.close()

f = open("test_labels.txt", "r", encoding="utf-8")
labels = []
for line in f:
    line = line.replace("\n", "")
    labels.append(line)
f.close()

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

features = features
labels = [labels_mapping[label] for label in labels]

features = np.array(features)
features = features / 128

x_test = features
y = labels
y_test = keras.utils.to_categorical(y, num_classes=classes)

model = keras.models.load_model("saved_model.h5")

loss, accuracy = model.evaluate(x_test, y_test)
print("Model  Loss: {}; Acc: {}".format(loss, accuracy))
predictions = model.predict_classes(x_test)
predictions = [labels_mapping_reverse[index] for index in predictions]
print(predictions)
prediction = "".join(predictions)

f = open("traducere.json", "w")
json.dump(predictions, f)
f.close()

actual = [labels_mapping_reverse[index] for index in y]
actual_str = "".join(actual)
print("Actual string:")
print(actual_str)
print("Predicted string:")
print(prediction)
