def base_to_decimal(number, base):
    decimal = 0
    power = 0
    while number > 0:
        digit = number % 10
        decimal += digit * (base ** power)
        number //= 10
        power += 1
    return decimal

x = 39
y=3

b10y=base_to_decimal(y,20)

b10x=base_to_decimal(x,20)

print(b10x,b10y,b10x%b10y)