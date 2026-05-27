from matplotlib import pyplot as plt
filename = "res/res_q1_bmitu.txt"
def read(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(line.strip().split(", "))
    return data

data = read(filename)
# X is the first column of data, Y is the last column of data
X = [int(x[0]) for x in data]
Y = [float(x[-1]) for x in data]
plt.plot(X, Y)
plt.xlabel("Number of result tuples")
plt.ylabel("Time (s)")
plt.title("Q1 REnum-BMITU")
plt.show()