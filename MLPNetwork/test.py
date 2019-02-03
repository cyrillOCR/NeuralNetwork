import json
import collections
import numpy as np
import keras
import time


# transforma o lista continand mai multe liste de numere intr-o singura lista doar cu numere
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


# fisierul din care se vor lua datele de test
file = "test_features.json"
# folosim flatten pe listele din input si le adaugam in lista features
features = []
f = open(file, "r")
a = json.load(f)
for nested_lists in a.values():
    flat_list = flatten(nested_lists)
    features.append(flat_list)
f.close()

# fisierul din care se vor lua label-urile pentru test
# test_labels.txt contine etichetele datelor de antrenare, fiecare pe cate un rand
# adaugam toate label-urile intr-o lista
f = open("test_labels.txt", "r", encoding="utf-8")
labels = []
for line in f:
    line = line.replace("\n", "")
    labels.append(line)
f.close()

# citim toate caracterele posibile care pot aparea in antrenare
# fisierul labelsset.txt contine toate caracterele distincte din datele de antrenare, fiecare pe cate o linie
f = open("labelsset.txt", "r", encoding="utf-8")
labels_set = []
for line in f:
    line = line.replace("\n", "")
    labels_set.append(line)
f.close()
# numarul de categorii pe care le va recunoaste reteaua = numarul de caractere distincte din datele de antrenare
classes = len(labels_set)

# construim doua dictionare: index_numeric: caracter si caracter: index_numeric
labels_mapping = dict()
labels_mapping_reverse = dict()
for index, label in enumerate(labels_set):
    labels_mapping[label] = index
    labels_mapping_reverse[index] = label

# inlocuim caracterele din labels cu indexurile lor numerice din dictionarul construit
labels = [labels_mapping[label] for label in labels]
# transformam features intr-un np array
features = np.array(features)
# normalizam trasaturile reducandu-le la intervalul [-1,1]
features = features / 128

# x_test va fi format din trasaturile normalizate
x_test = features
y = labels
# y_test va fi format din cate un vector categoric pentru fiecare valoare din labels
y_test = keras.utils.to_categorical(y, num_classes=classes)

startTime = time.time()
# incarcam modelul antrenat
model = keras.models.load_model("saved_model.h5")
# evaluam metricile modelului pe datele de test
loss, accuracy = model.evaluate(x_test, y_test)
print("Model  Loss: {}; Acc: {}".format(loss, accuracy))
# prezicem clasele carora apartin caracterele
predictions = model.predict_classes(x_test)
# inlocuim indexul numeric al categoriei cu caracterul aferent din dictionarul construit
predictions = [labels_mapping_reverse[index] for index in predictions]
print(time.time() - startTime)
# salvam lista de litere in fisierul traducere.json
f = open("traducere.json", "w")
json.dump(predictions, f)
f.close()
