#use a while loop to print "hello" x 3

l=3
while l>0:
    print("hello\n")
    l -=1

#________


#as long as the user does not give the right input/
#as long as the user does not do somehting to exit
#while True -> break

#while True:
    
#________

#For _ in range n

for i in range (10):
    if (i==5):
        break
    print("hello")

#break vs continue

#___________
# lists:
    #allow to mix types
# tuples: more difficult to modify parameters
# dictionaries
# arrays
# range object: range of numbers
    # range(start,end) ; range(end)
    # also has parameter "step", to choose of how many units you jump each step

##r = range(10)
#print(r[0]) 
#print(r[10]) #range error

#L= [1,2,3,4]
#for i in range():
#    print(L[i]) #in this case, the iteration relies on L, so i don't need to put range parameter

#len() takes a list, return the number of elements in the list (int)

M=[["a", "b"], 2,3,4]
#L[0] > class == list
#L[0][0] == 'a'

print(len(M))