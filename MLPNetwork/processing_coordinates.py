import json

# f = open("coord7.json", "r")
# g = open("normal.txt","w")
# a = json.load(f)
# list = a.get("coords")
# print(len(list))
# for el in list:
#     for e in el:
#         g.write(str(e))
#         g.write(" ")
#     g.write("\n")


g = open("pg10.json", "r")
file = "data10.json"
f = open(file, "r")
a = json.load(f)
b = json.load(g)
a_keys = set(a.keys())
b_keys = set(b.keys())
print("Nou: {}".format(a.keys()))
print("Vechi: {}".format(b.keys()))
print("Nou: {}".format(len(a.keys())))
print("Vechi: {}".format(len(b.keys())))
dif = a_keys - b_keys
print("Nou - vechi: {} ".format(len(dif)))

# dif = [16,40,46,53,54,151,274,332,349,359,369,399,408,459]
for el in dif:
    del a[str(el)]

f.close()
f = open(file, "w")
json.dump(a, f)
f.close()
print(len(a))
