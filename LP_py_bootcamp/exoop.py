# OBJECT ORIENTED PROGRAMMING

# programming paradigms:
# procedural (C)
# functional
# OOP (Java,C++)

# types: tuple, classes, 

# class vs object: 
# object: instance of a class
# class is a type
# defining a class -> instruction that is executed

# mother class
class Human:
    ...

#child class
class Student(Human):
    ...

# an object has attributes and methods
# an object is created through an INSTANCE METHOD
# object_name.method_name()
# attribute is a "characteristic"

#how to instantiate the objects:
class Person:
    def __init__(self, value=25):

        # name of the attribute = value for this naem
        self.age = value

    def aging(self):  #self is always mandatory
        self.age = self.age+1  #when calling method 'aging' , age will be increased by 1

# argument 1 of init is the object of the class, called 'self' by convention

# instance variable vs class variable

# 'print' is overloaded in int,float,str,...


# tuples: t = (a,b)

def main():
    person = Person(27)
    print(person.age)

    person2 = Person()
    print(person2.age) # in this case age will be 25, bc it's the default value