import json
import predict
import time

file = "test_features.json"
f = open(file, "r")
a = json.load(f)
f.close()

start_time = time.time()
predict.predict(a)
print(time.time()-start_time)