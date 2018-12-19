input_file = open("input.txt", "r", encoding="utf-8")
content = input_file.read()
content = content.replace(" ", "").replace("\n", "")
labels_file = open("labels.txt", "w", encoding="utf-8")
for character in content:
    labels_file.write(character + "\n")
