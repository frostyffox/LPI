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
from sorted_f import get_house, get_student

#  BROWSING A FILE LINE BY LINE
L = []
with open("students.csv") as file:
    next(file) #skip header
    for i in file: #i == string
        L.append(i.strip())

#__sorting the file
print("Sorting aplhabetically")
print("-"*40)
for i in sorted(L):
    print(i)
print("-" * 40)
#__printing the sorted list
print("Sorting by house:")
for i in L:
    print(i)
print("-"* 40)

#if i want to sort using not the 1st column, i import the data into a dictionary
dict = {}
L = []
with open("students.csv") as file:
    next(file) #skip header
    for i in file:
        #row = i.strip().split(",")
        name, col2 = i.strip().split(",") #this way 'name' and 'col2' cannot be modified

        #dict[key] = value -> because the dict already exists
        dict[name] = col2
    #print(f"Dictionary: {dict}")
print("Dict sorted by key:")
print("-"*40)
for i in sorted(dict):
    #print(i,dict[i])
    print(f"{i:15}|{dict[i]}") #as line above but formatted
print("\n*****************")

#if you want to have control on which column you use to sort -> need list of dicts
# every line of the csv is a dict
# the whole file is a list of dict -> L
# L[0]{"name"} is first line of the file, column 'name'
with open("students.csv") as file:
    next(file) #skip header
    print("Making a list of dictionaries first")
    
    for l in file:
        name, house= l.strip().split(",")
        L.append({"name": name, "house":house})
    print(L)

# for each dictionary d, inside list L, 
# use the value in "house" column as sorting key
print("\nSorted by house")
print("-"*40)

#____option w lambda_____
#for i in sorted(L, key=lambda d: d["house"] ):
#    print(i["name"], i["house"])

L1 = sorted(L,key=get_student)
L2 = sorted(L, key=get_house)

print(L2)
