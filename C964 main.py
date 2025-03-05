import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model

df = pd.read_csv('.venv/dataset')

# print("I got here 1")
# print(df)
# df.hist()
# pyplot.show()
# print("I got here 2")
# scatter_matrix(df)

#mylog_model = linear_model.LogisticRegression()
y = df.values[:,5]
x = df.values[:,2]


# print("I got here 3")
print("Printing Y's: ")
print(y)
print("Printing X's: ")
print(x)
