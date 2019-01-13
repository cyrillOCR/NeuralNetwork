import json
import predict
import time

file = "data10full.json"
f = open(file, "r")
a = json.load(f)
f.close()

start_time = time.time()
predictions = predict.predict(a)
print(predictions)
print(time.time() - start_time)
prediction = "".join(predictions)
print(prediction)

file = open("replace.txt","r",encoding="utf-8")
for line in file:
    line = line.strip()
    wrong_char, good_char = line.split("-")
    prediction = prediction.replace(wrong_char,good_char)

print(prediction)