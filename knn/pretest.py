import struct
from numpy import *

print(zeros(2, 3))

with open(r"..\data\knn\t10k-images.idx3-ubyte", "rb") as f:
    a = f.read(4)
    (b,) = struct.unpack("i", a)
    print(b)
    print(hex(b))

    f.seek(16)
    for r in range(28):
        for c in range(28):
            pixel = f.read(1)
            if (ord(pixel) != 0):
                print("*", end="")
            else:
                print(" ", end="")
        print("")
    f.close()
with open(r"..\data\knn\t10k-labels.idx1-ubyte", "rb") as f:
    f.seek(8)
    print(ord(f.read(1)))
    f.close()
