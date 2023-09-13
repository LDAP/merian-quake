import sys
from math import exp

if (len(sys.argv) == 1):
    d = int(input("enter particle size in µm: "))
else:
    d = int(sys.argv[1])

g = exp(-2.20679 / (d + 3.91029) - 0.428934)
a = exp(3.62489 - 8.29288 / (d + 5.52825))

print(f"α = {a}, g = {g}")
