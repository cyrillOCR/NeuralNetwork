import json
import collections
import numpy as np
import keras


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
f.close()

f = open("labelsset.txt", "r", encoding="utf-8")
labels_set= []
for line in f:
    line = line.replace("\n", "")
    labels_set.append(line)
f.close()


labels_mapping = dict()
labels_mapping_reverse = dict()
for index, label in enumerate(labels_set):
    labels_mapping[label] = index
    labels_mapping_reverse[index] = label

features = features[0:446]
labels = [labels_mapping[label] for label in labels]

features = np.array(features)
features = features / 128

y = keras.utils.to_categorical(labels[0:50], num_classes=33)

model = keras.models.load_model("saved_model.h5")

loss, accuracy = model.evaluate(features[0:50],y)
print("Model  Loss: {}; Acc: {}".format(loss, accuracy))
predictions = model.predict_classes(features[0:50])
predictions = [labels_mapping_reverse[index] for index in predictions]
print(predictions)
prediction = "".join(predictions)

f = open("traducere.json","w")
json.dump(predictions,f)
f.close()

actual = [labels_mapping_reverse[index] for index in labels[0:50]]
actual_str = "".join(actual)
print("Actual string:")
print(actual_str)
print("Predicted string:")
print(prediction)
