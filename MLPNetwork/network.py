import json
import keras
import time
import collections
import numpy as np
from keras.layers import InputLayer, Dense, Dropout


# transforma o lista continand mai multe liste de numere intr-o singura lista doar cu numere
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


# numarul de neuroni de pe stratul de input - numarul de trasaturi pentru un caracter
input_shape = 454
# fisierul din care se vor lua datele de antrenament
training_file = "training_features.json"

# folosim flatten pe listele din input si le adaugam in lista features
f = open(training_file, "r")
a = json.load(f)
features = []
for nested_lists in a.values():
    flat_list = flatten(nested_lists)
    features.append(flat_list)
f.close()

# fisierul din care se vor lua label-urile pentru antrenament
# training_labels.txt contine etichetele datelor de antrenare, fiecare pe cate un rand
# adaugam toate label-urile intr-o lista
training_file = "training_labels.txt"
labels = []
f = open(training_file, "r", encoding="utf-8")
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
# initializam x_train cu trasaturile normalizate
x_train = features
# initializam y_train cu cate un vector categoric(1 pe pozitia index-ului, 0 pe restul) pentru valorile din labels
y_train = keras.utils.to_categorical(labels, num_classes=classes)

# construim un model secvential de rn
model = keras.models.Sequential()
# initializam weight-urile si bias-urile
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
# adaugam layer-ul de input modelului
model.add(InputLayer(input_shape=(input_shape,)))
# adaugam layer-ul hidden, cu 500 de neuroni relu
model.add(Dense(500, input_shape=(input_shape,), activation="relu"))
# adaugam layer-ul de output
model.add(Dense(classes, input_shape=(input_shape,), activation="softmax"))

startTime = time.time()
# compilam modelul
model.compile(optimizer=keras.optimizers.Adam(),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
# antrenam modelul
model.fit(x_train, y_train, epochs=7, batch_size=10)
# salvam modelul antrenat in fisierul saved_model.h5
keras.models.save_model(model, "saved_model.h5", overwrite=True, include_optimizer=True)
print(time.time() - startTime)
