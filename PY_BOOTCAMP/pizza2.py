# goal: find the most rewarding pizza

import csv

# open the file
with open("pizza.csv", newline='') as pi:

    # go through the lines
    read = csv.reader(pi)
    ratio = 0
    which_pizza = ""
    next(read, None) #skip header

    # extract the small and large prices
    for line in read:
        name = line[0]
        # assign to ratio variable
        small = float(line[1].lstrip("$"))
        large = float(line[2].lstrip("$"))

        new_ratio = large/small
        print(f"New ratio: {new_ratio:.2f}$, for {name}")

        # confront with previous value
        if new_ratio > ratio:
            # if new ratio > old ratio: change ratio value
            ratio = new_ratio
            print(f"updated value: {ratio:.2f}")
            which_pizza = name
    
    print(f"The most rewarding pizza is {which_pizza}")
    # return line[0] of highest ratio

