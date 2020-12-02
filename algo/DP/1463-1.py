import sys
num = int(sys.stdin.readline())
cache = [0] *  (num + 1)

for i in range(2, num+1):
    value = -1

    # divide 2
    if i % 2 == 0:
        value = cache[int(i/2)]
    # divide 3
    if i % 3 == 0:
        if value == -1:
            value = cache[int(i/3)]
        else:
            value = min(value, cache[int(i/3)])
    # minus 1
    if value == -1:
        value = cache[i - 1]
    else:
        value = min(cache[i - 1], value)

    cache[i] = value + 1  

print(cache[num])