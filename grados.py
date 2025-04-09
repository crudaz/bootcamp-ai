# python function to convert grades celsius to fahrenheit
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# python function to convert grades fahrenheit to celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# print results
celsius = 25
fahrenheit = 77

print(f"{celsius}°C is equal to {celsius_to_fahrenheit(celsius)}°F")
print(f"{fahrenheit}°F is equal to {fahrenheit_to_celsius(fahrenheit)}°C")
