import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import math
data=pd.read_csv("weight-height.csv")
heights=data["Height"].to_numpy().reshape(-1,1)
weights=data["Weight"].to_numpy()
regressor=LinearRegression()
regressor.fit(heights,weights)
pred=regressor.predict(heights)
rmse=math.sqrt(mean_squared_error(weights,pred))
r2=r2_score(weights,pred)
print("Linear model -> weight = {:.4f}*height + {:.4f}".format(regressor.coef_[0],regressor.intercept_))
print("rmse:",rmse," r2:",r2)
h_line=np.linspace(heights.min(),heights.max(),300).reshape(-1,1)
w_line=regressor.predict(h_line)
plt.scatter(heights,weights,s=5,alpha=0.4)
plt.plot(h_line,w_line,color="black")
plt.savefig("vB_reg.png")
print("Image saved as vB_reg.png")
plt.show()
