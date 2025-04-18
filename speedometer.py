import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import math

#Read file

dt = 0.144

f = None
try:
    f = open("chaos1.txt","r") #à changer
except:
    print("Result file not found.")
    exit(1)

text = f.readlines()

f.close()

data = []

for i in range(2,len(text)):
    row = text[i].split('  ')
    data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), 
row[4].strip() == "True"])
data = np.array(data)

fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [1, 1, 1, 14]}, figsize=(7, 10))

ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')

ax[3].set_xlim(-6, 6)
ax[3].set_ylim(-6, 6)
q = ax[3].quiver(0, 0, 1, 1)

ax[0].set_xlim(0.4, 1.4)
ax[1].set_xlim(0.4, 1.4)
ax[2].set_xlim(0.4, 1.4)
text_x = ax[0].text(.5, .5, '', fontsize=15)
text_y = ax[1].text(.5, .5, '', fontsize=15)
text_n = ax[2].text(.5, .5, '', fontsize=15)

def updatefig(frame):
    text_x.set_text("Vitesse horizontale : " + str(data[frame][2]) + " [m/s]")
    text_y.set_text("Vitesse verticale : " + str(data[frame][3]) + " [m/s]")
    text_n.set_text("Vitesse absolue : " + "{:.7f}".format(math.sqrt((data[frame][2])**2 + (data[frame][3])**2)) + " [m/s]")
    q.set_UVC(data[frame][2], data[frame][3])
    if(data[frame][4]):
        text_x.set_c("green")
        text_y.set_c("green")
        text_n.set_c("green")
        q.set(color="green")
    else:
        text_x.set_c("red")
        text_y.set_c("red")
        text_n.set_c("red")
        q.set(color="red")
    return text_x, text_y, text_n, q

anim = FuncAnimation(fig, updatefig, frames=len(data), blit=True, interval = dt*1000)
anim.save("speed_chaos1.gif",writer=PillowWriter(fps=1/dt))
plt.show()


