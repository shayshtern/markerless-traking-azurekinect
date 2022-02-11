import time
import timeit

import pandas as pd
import numpy as np

# excel_file = 'kinectskeletontest.xlsx'
# df = pd.read_excel(excel_file)

# print(df.iloc[2:10, 2:5], '\n')  # Neck X1,Y1,Z1
# print(df.iloc[5, 2], '\n')  # first value in the dataframe
#
# df.iat[5, 2] = 10  # set value of a cell
#
# print(df.iloc[5, 2], "\n")  # first value in the dataframe
#
# print(df.iloc[2:10, 2:5], '\n')  # Neck X1,Y1,Z1
#
# df.to_excel("output.xlsx", header=False, index=False)
# markers = df.iloc[2, :]  # The row with the marker's names
#
# print("markers:", '\n', markers, '\n')
# print(type(markers), '\n')

# print(markers.where(markers == "Neck"), inplace=)
a1 = np.array([1, 2, 3])
a2 = np.array([5, 6, 7])
alpha = 180
# print(np.cos(alpha * np.pi / 180))


import time
start = time.time()
time.sleep(1.5)
stop = time.time()
print("The time of the run:", stop - start)

print(np.add(10, 10))