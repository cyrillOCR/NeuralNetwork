import json
#
# f = open("b23.txt", "r")
# g = open("coord_normal.txt","w")
# h = open("coord_modificat.txt","w")
# a = json.load(f)
# list = a.get("coords")
# print(len(list))
# for el in list:
#     for e in el:
#         g.write(str(e))
#         g.write(" ")
#         h.write(str(e))
#         h.write(" ")
#     g.write("\n")
#     h.write("\n")


file = "data23.json"
f = open(file, "r")
a = json.load(f)

dif = [
    0,
    3,
    101,
    102,
    141
]

for el in dif:
    del a[str(el)]

f.close()
f = open(file, "w")
json.dump(a, f)
f.close()
print(len(a))
