import os
import pandas as pd
import random


image_files = []
os.chdir("images")
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("images/" + filename)
os.chdir("..")

random.shuffle(image_files)
print(len(image_files))
tra = 0;
val = 0;
tes = 0;
with open("images/train.txt", "w") as outfile:
    for image in range(1500):
        outfile.write(image_files[image])
        outfile.write("\n")
        tra +=1
    outfile.close()

with open("images/validation.txt", "w") as outfile:
    for image in range(1500,1900,1):
        outfile.write(image_files[image])
        outfile.write("\n")
        val +=1
    outfile.close()

with open("images/test.txt", "w") as outfile:
    for image in range(1900, 2000, 1):
        outfile.write(image_files[image])
        outfile.write("\n")
        tes += 1
    outfile.close()
print("image for testing: " + str(tes))
print("image for training: " + str(tra))
print("image for validation: " + str(val))
os.chdir("..")
