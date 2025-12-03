# _____ FILES _______


#import something to handle file format 
import csv

all_lines = []

# open file
with open('Name.csv', newline="") as csvf:
# go through the lines -> use a reader object in loop
    obj = csv.reader(csvf)
    for line in obj:


        # store all lines in a list
        all_lines.append(line)
        
