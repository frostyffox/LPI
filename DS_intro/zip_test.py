a = ["France", "Germany", "Denmark", "Iceland", "Spain", "Italy"]
b = [12.5, 14.2, 10.1, None, 16.8, 20.0]

x = zip(a,b)
print(list(x))

zip_dict = dict(x)

print(zip_dict)