# mode      access and go through   modify                      short
# reading       yes                 no                              r
# writing       yes                 yes (overwriting)               w
# append        yes                 yes (without overwriting)       a

#file = open(filename, mode="r")
# fileID: output of the open fn
# 'file' is a pointer to the object handled by open
# filename is a string

# methods
# write(str)
# read()
# close() -> can be omitted if: "with open(path) as file":\n,tab, block

# attributes

import csv

#  BROWSING A FILE LINE BY LINE
L = []
with open("students.csv") as file:
    for i in file: #i == string
        L.append(i.strip())

#__sorting the file
for i in sorted(L):
    print(i)
print("Sorting done\n*********************")
#__printing the sorted list
for i in L:
    print(i)
print("*********************\nNow sorting by column 2")

#if i want to sort using not the 1st column, i import the data into a dictionary
dict = {}
L = []
with open("students.csv") as file:
    for i in file:
        #row = i.strip().split(",")
        name, col2 = i.strip().split(",") #this way 'name' and 'col2' cannot be modified

        #dict[key] = value -> because the dict already exists
        dict[name] = col2
    #print(f"Dictionary: {dict}")
print("Dict sorted by key:")
for i in sorted(dict):
    print(i,dict[i])
print("*****************")

#if you want to have control on which column you use to sort -> need list of dicts
# every line of the csv is a dict
# the whole file is a list of dict -> L
# L[0]{"name"} is first line of the file, column 'name'
with open("students.csv") as file:
    print("Making a list of dictionaries first")
    for l in file:
        name, house= l.strip().split(",")
        L.append({"name": name, "house":house})
    print(L)

# for each dictionary d, inside list L, 
# use the value in "house" column as sorting key
for i in sorted(L, key=lambda d: d["house"] ):
    print(i["name"], i["house"])

