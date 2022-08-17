import os
import pandas as pd
import random


image_files = []
os.chdir(os.path.join("data", "obj"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("data/obj/" + filename)
os.chdir("..")

random.shuffle(image_files)
print(len(image_files))
tra = 0;
val = 0;
tes = 0;
with open("/content/drive/MyDrive/training yolov5/train.txt", "w") as outfile:
    for image in range(24000):
        outfile.write(image_files[image])
        outfile.write("\n")
        tra +=1
    outfile.close()

with open("/content/drive/MyDrive/training yolov5/validation.txt", "w") as outfile:
    for image in range(24000,25000,1):
        outfile.write(image_files[image])
        outfile.write("\n")
        val +=1
    outfile.close()

with open("/content/drive/MyDrive/training yolov5/test.txt", "w") as outfile:
    for image in range(25000, 35000, 1):
        outfile.write(image_files[image])
        outfile.write("\n")
        tes += 1
    outfile.close()
print("image for testing: " + str(tes))
print("image for training: " + str(tra))
print("image for validation: " + str(val))
os.chdir("..")