# input_file = open("24tradus.txt", "r", encoding="utf-8")
# content = input_file.read()
# content = content.replace(" ", "").replace("\n", "")
# labels_file = open("labels24.txt", "w", encoding="utf-8")
# for character in content:
#     labels_file.write(character + "\n")


files = ["labels3.txt","labels4.txt","labels5.txt","labels6.txt","labels7.txt","labels8.txt","labels10.txt","labels20.txt","labels21.txt",
         "labels22.txt","labels23.txt","labels24.txt"]
labels = []
for file in files:
    f = open(file, "r", encoding="utf-8")
    for line in f:
        line = line.replace("\n", "")
        labels.append(line)
    f.close()

labels_set = set(labels)
l_l = list(labels_set)
l_l.sort()
f = open("labelsset.txt","w",encoding="utf-8")
for el in l_l:
    f.write(el)
    f.write("\n")