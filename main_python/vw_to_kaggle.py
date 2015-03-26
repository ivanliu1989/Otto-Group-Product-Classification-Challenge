__author__ = 'philipmargolis'
import math

def zygmoid(x):
    return 1 / (1 + math.exp(-x))

with open("submissions/vw_3.csv","wb") as outfile:
    outfile.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
    for line in open("oaa.rawp.txt"):
        if line[0] == "0": #skip first 9 lines of each prediction
            continue
        #process the 10th line, which does not begin with a 0
        row = line.strip().split(" ")
        outfile.write("%s,"%row[9]) #write Id
        for m in range(8):
            outfile.write("%f,"%(zygmoid(float(row[m][2:])))) #write prediction for first 8 classes
        outfile.write("%f\n"%(zygmoid(float(row[8][2:])))) #write prediction for 9th class
