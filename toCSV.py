import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv

# col_name = ['epoch']
#
# for i in range(0,10,1):
#     records = [i]
#
#     df = pd.DataFrame(columns=col_name, data=records)
#
#     df.to_csv("./adam.csv", encoding='utf-8', index=False)
#
# print("all done")

list_res=[]
list_strip_names=[1,2,3]
list_one_counts=[4,5,6]
list_match=[7,8,9]

for i in range(len(list_one_counts)):
    list_res.append([list_strip_names[i],list_one_counts[i],list_match[i]])

column_name = ['name', 'time' , 'hh']
csv_name='nth.csv'
xml_df = pd.DataFrame(list_res, columns=column_name)
xml_df.to_csv(csv_name, index=None)

plt.rcParams['font.sans-serif']=['SimHei']

plt.rcParams['axes.unicode_minus']=False


# x=[]
# y=[]
# with open('nth.csv','r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x.append(row[0])
#         print("x:",x)
#         y.append(row[1])
#         print("y:",y)
#
# plt.plot(x,y,label='minishuju')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('load data')
# plt.legend()
# plt.show()
