celsius_values = [12.5, 14.2, 10.1, None, 16.8, 20.0]
country_names = ["France", "Germany", "Denmark", "Iceland", "Spain", "Italy"]

# 1.2 Convert non-missing values
celsius_float_values = []
farenheit_vals = []
for c in celsius_values:
    if c is not None: 
        celsius_float_values.append(float(c))

        in_far = c * 9/5 + 32
        farenheit_vals.append(in_far)
    

    # Formula to get Fahrenheit value from Celsius value : f = c * 9/5 + 32

# 1.3 Compute mean ignoring missing values
sum_far = 0
if len(farenheit_vals) > 0:     
#or
    sum_far = sum(farenheit_vals)


    mean_vals = sum_far/len(farenheit_vals)
    print(f"Mean: {mean_vals}")
else: 
    print("No values provided")

def classify_temp(c):
    if len(celsius_values) == 0:
        print("Hehe no values in this list")
    if c != None:
 
        if c< 5:
            return 'cold'
        elif 5<= c < 15:
            return 'mild'
        elif c >= 15:
            return 'warm'
        else:
            print("Not a valid temperature value!")
    else:
        return None

labels = []
for c in celsius_values:
    labels.append(classify_temp(c))
print(labels)

# 3.1 List comprehension
fahrenheit_comp = [item for item in farenheit_vals if item is not None]

# basic method
# fahrenheit_comp =[]
# for i in farenheit_vals:
#   if i != None:
#       fahrenheit_comp.append(i)

# 3.2 zip() function -> returns a zip object
# dict name: country_to_celsius 

#remove None values
celsius_comp = [item for item in celsius_values if item is not None]

# zipping and converting in dict
country_celsius = dict(zip(country_names, celsius_comp))
#print(list(x))
#zip_dict = dict(x)

# 3.3 items() function
country_to_label = {k: classify_temp(v) for k,v in country_celsius.items()}

print(f"Country to label: {country_to_label}")
