#from cs50 course exercise set

import random
import sys 

# get random number
def get_rn():
    while True:
        try:
            #prompt the user for a positive integer
            pi = int(input("Choose a level between 1 and 100: "))
            #if input < 0: reprompt
            if pi <= 0:
                continue
            
            rn = random.randint(1,pi)
            print(f"Your input: {pi}.\nRandom number: {rn}\n")
             #random number 
             #rn = random.randint(1,pi)
            return pi, rn
        except ValueError:
            print("Input a positive integer!")

   

# pi: first input
# rn: random number from get_rn
def guess(pi,rn):
    while True:
        try:
            gs = int(input(f"Guess the number beetween 1 and {pi}\n"))
            #prompt the user for a guess "gs"

            if gs < 1 or gs > pi:
                continue
            # if gs < rn : "Too small", prompt again
            if gs < rn:
                print("Too small!")
                
            # if gs > rn : "Too big", prompt again
            elif gs > rn:
                print("Too big!")
                
            # if gs == rn : Just right, exit
            elif gs == rn:
                print("Just right!")
                break
        except ValueError:
            print("Choose a valid positive int")
            

def main():
    pi, rn= get_rn()
    if rn < 1 or rn > 100:
        sys.exit("The value generated contains a mistake")
    guess(pi, rn)


if __name__ == "__main__":
    main()
    
