import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

table=pd.read_csv("https://academy.dqlab.id/dataset/penduduk_gender_head.csv")
table.head()

x_label=table['NAMA KELURAHAN']
x=np.arange(len(x_label))
height=table['LAKI-LAKI WNI']
plt.bar(x,height)
plt.xticks(x, x_label, rotation=40)
plt.show()