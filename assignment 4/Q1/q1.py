import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)
sample_counts=[500,1000,2000,5000,10000,15000,20000,50000,100000]
theory={2:1/36,3:2/36,4:3/36,5:4/36,6:5/36,7:6/36,8:5/36,9:4/36,10:3/36,11:2/36,12:1/36}
x_axis=np.array(list(theory.keys()))
y_axis=np.array(list(theory.values()))
for count in sample_counts:
    dice_one=np.random.randint(1,7,size=count)
    dice_two=np.random.randint(1,7,size=count)
    sums=dice_one+dice_two
    hist,bins=np.histogram(sums,range(2,14))
    freq=hist/count
    plt.figure()
    plt.bar(bins[:-1],freq,color="skyblue",label="simulated")
    plt.plot(x_axis,y_axis,color="red",marker="o",label="expected")
    plt.title("Samples: "+str(count))
    plt.legend()
    filename=f'vB_hist_{count}.png'
    plt.savefig(filename)
    print("Image saved as",filename)
    plt.close()
print("finished B ex1")



# answer 4: Small n shows random bumps, large n makes the shape stable and close to theory.
# answer 5: Extreme deviations shrink with more trials, data returns toward the central pattern.
