import statistics as st
import csv
import numpy
import pandas
import os

dirs = [d for d in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(),d))]

# Discover number of rows
dir = dirs[0]
path = dir
fileIn = open(path+"/accuracy/ibk_accuracy.txt","r")
rowCounter = 0
for row in fileIn.readlines():
    if row.startswith("*"):
        rowCounter +=1
m = rowCounter

# Init
accTra = []
accTes = []
n = 12
fullTable = [[0 for col in range(n)] for row in range(m)]
tags = ["sco","lin","ibk"]
row_labels = []
col_labels = []

for dir in dirs:
    path = dir
    for tag in tags:
        fileIn = open(path+"/accuracy/"+str(tag)+"_accuracy.txt","r")
        rowCounter = -1

        # Start parsing txt
        for row in fileIn.readlines():
            if row.startswith("*"):
                row_labels.append(row.split(" ")[2].split("-")[0])
                rowCounter+=1
            else:
                if not row.startswith("#") and not row.startswith("m"):
                    cells = row.split("\t\t")
                    runId = int(cells.pop(0))
                    accTra.append(float(cells.pop(0)))
                    accTes.append(float(cells.pop(0).split("\n")[0]))
                    if runId == 10:
                        index = int(tags.index(tag))
                        fullTable[rowCounter][index*2] = st.mean(accTra)
                        fullTable[rowCounter][index*2+1] = st.stdev(accTra)
                        fullTable[rowCounter][index*2+6] = st.mean(accTes)
                        fullTable[rowCounter][index*2+7] = st.stdev(accTes)
                        accTes.clear()
                        accTra.clear()

    # Write mean and stDev csv
    col_labels = ["tra_sco_mean","tra_sco_stdDev","tra_lin_mean","tra_lin_stdDev","tra_ibk_mean",
              "tra_ibk_stdDev","tst_sco_mean","tst_sco_stdDev","tst_lin_mean","tst_lin_stdDev",
              "tst_ibk_mean", "tst_ibk_stdDev",]
    row_labels = row_labels[0:m]
    df = pandas.DataFrame(numpy.asmatrix(fullTable), index=numpy.asarray(row_labels), columns=numpy.asarray(col_labels))
    df.to_csv(path+"_accuracy_table.csv", sep=',', encoding='utf-8')

    # Write testing csv
    columns = [6, 8, 10]
    col_labels = ["sco","lin","ibk"]
    df = pandas.DataFrame(numpy.asmatrix(fullTable)[:,columns],index=numpy.asarray(row_labels), columns=numpy.asarray(col_labels))
    df.to_csv(path+"_accuracy_test_table.csv", sep=',', encoding='utf-8')