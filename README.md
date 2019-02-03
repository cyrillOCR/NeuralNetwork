# NeuralNetwork

Cerinte:
- python 3.4,3.5 sau 3.6
- numpy(pip install numpy)
- keras(citeste instructiunile de instalare din Keras.pdf)

Fisiere necesare:
- training_features.json(contine datele de antrenament, obtinute de la modulul de extragere trasaturi, in format json)
- test_features.json(contine datele de test, obtinute de la modulul de extragere trasaturi, in format json)
- training_labels.txt(contine etichetele de antrenament, fiecare caracter pe cate un rand)
- test_labels.txt(contine etichetele de test, fiecare caracter pe cate un rand)
- labelsset.txt(contine toate caracterele distincte din datele de antrenament si test, fiecare pe cate un rand) 
		labelesset.txt va influenta numarul de categorii(numarul de neuroni de output) la antrenarea retelei

Fisiere ce se vor genera la rularea codului:
- saved_model.h5, la rularea network.py -> contine modelul antrenat
- traducere.json, la rularea test.py -> contine lista de litere recunoscute pe datele de test