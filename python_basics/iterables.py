# good description of the iterables here:
# https://www.pythonlikeyoumeanit.com/Module2_EssentialsOfPython/Iterables.html


# list
myfirstlist = [1, 'st', 6.0, [2.0,3.0] ]
mysecondlist = [4 , 3, 7, 8]

# myfirstlist.append(9.0)

for index,element in enumerate(myfirstlist):
    print(index)
    print(element)


for first_element,second_element in zip(myfirstlist, mysecondlist):
    print(first_element)
    print(second_element)


#mylist[2] = 5.0

#print(mylist[3][0])

# tuple

# mytuple = (2, 'str', 8.0, [3.0,5.0] )

#mytuple[2] = 5.0

#print(mylist[0])

# dictionary

mydict = {'a':2, 5:3, 'c':6 }

print(mydict[5])

#for item in mydict.items():
#    print(item[0])
#    print(item[1])

for (key,element) in mydict.items():
    print(key)
    print(element)

# print(list(mydict.values()))

# for index, element in zip(mydict.keys(), mydict.values()):
#    print(index)
#    print(element)

