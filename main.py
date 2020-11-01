import numpy as np
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def judge(x):
    x=1 if x>=1.5 else 0
    return x

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

a=np.array([[1,2],[1,3]])
print(np.mean(a,axis=0))

x=np.array([1,2])
z=np.array([[1,1],[1,2]])
print(x.dot(z)**2)

print(np.vectorize(judge)(x))
