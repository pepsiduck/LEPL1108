import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

#Read file

f = None
try:
    f = open("resultats/chaos2.txt","r") 
except:
    print("Result file not found.")
    exit(1)

text = f.readlines()

data = []

for i in range(2,len(text)):
    row = text[i].split('  ')
    data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), 
row[4].strip() == "True"])
data = np.array(data)

# Simulation parameters

dt = 0.144  # Time step
steps = len(data)  # Number of time steps
var_u = 0.4 # Variance of the process noise
var_y = 0.3 # Variance of the measurement noise  :  to be determined

# Generate measurements (multistatic ranges)
measurements = np.zeros((steps, 4))

for i in range(steps):
    #measurements[i][0:2] = true_pos[i] + normal(0, var_y, 2) # Measurement noise
    #measurements[i][2:4] = true_vel[i] + normal(0, var_y, 2)  # Measurement noise


    if i != 0 and data[i][4] == 0:
        measurements[i][0] = measurements[gi][0]
        measurements[i][1] = measurements[gi][1]
        measurements[i][2] = measurements[gi][2]
        measurements[i][3] = measurements[gi][3]
    else:
        gi = i
        measurements[i][0] = data[i][0]
        measurements[i][1] = data[i][1]
        measurements[i][2] = data[i][2]
        measurements[i][3] = data[i][3]


# Kalman Filter setup
kf = KalmanFilter(dim_x=4, dim_z=4)
kf.F = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

kf.H = np.eye(4)
#kf.H = np.zeros((2, 4))
#kf.H[0:2,2:4] = np.eye(2) 
kf.R = np.eye(4) * var_y 
kf.Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                 [0, dt**4/4, 0, dt**3/2],
                 [dt**3/2, 0, dt**2, 0],
                 [0, dt**3/2, 0, dt**2]]) * var_u * 2

kf.x = np.array([data[0][0], data[0][1], data[0][2], data[0][3]])
estimates = np.zeros((steps, 4))

# Kalman Filter loop
for i in range(steps):
    kf.predict()
    kf.update(measurements[i])
    estimates[i] = kf.x


fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('kalman filter trajectory showcase')
ax.grid()

#true_line, = ax.plot([], [], label='True trajectory', color='blue')
#meas_scatter = ax.scatter([], [], label='Measurements', color='orange', alpha=0.5)
mesu_scatter_good = ax.scatter([], [], label='Good Measurements', color='green', alpha=0.5)
mesu_scatter_wrong = ax.scatter([], [], label='Wrong Measurements', color='red', alpha=0.5)
est_line, = ax.plot([], [], label='Kalman Estimate trajectory', color='green')
ax.legend()




velocity_arrow = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1.5, label='Velocity', color='darkblue', width=0.005)
velocity_text = ax.text(-6.5, 10, '', fontsize=12, color='darkblue')


def update(frame):

    good = []
    wrong = []

    for i in range(0, frame):
        if data[i][4] == 1:
            good.append(np.array(data[i][0:2]))
        else:
            wrong.append(np.array(data[i][0:2]))


    if len(good) != 0:
        mesu_scatter_good.set_offsets(np.array(good))
    else:
        mesu_scatter_good.set_offsets(np.array(data[:0, 0:2]))


    if len(wrong) != 0:
        mesu_scatter_wrong.set_offsets(np.array(wrong))
    else:
        mesu_scatter_wrong.set_offsets(np.array(data[:0, 0:2]))


    est_line.set_data(estimates[:frame, 0], estimates[:frame, 1])


    if frame > 0:
        x_pos = estimates[frame - 1, 0]  
        y_pos = estimates[frame - 1, 1]  
        x_vel = estimates[frame - 1, 2]  
        y_vel = estimates[frame - 1, 3]  
        velocity_arrow.set_offsets([[x_pos, y_pos]])
        velocity_arrow.set_UVC(x_vel, y_vel)
        abs_velocity = np.sqrt(x_vel**2 + y_vel**2)
        velocity_text.set_text(f'Velocity: {abs_velocity:.2f} m/s')
    else:
        velocity_arrow.set_offsets([[0, 0]])
        velocity_arrow.set_UVC(0, 0)
        velocity_text.set_text('Velocity: 0.00 m/s')


    ax.set_xlim(-7, 7)
    ax.set_ylim(0, 11)

    return mesu_scatter_good, mesu_scatter_wrong, est_line, velocity_arrow, velocity_text


anim = FuncAnimation(fig, update, frames=steps, interval=100, blit=True)
anim.save('chaos2_new.gif', writer=PillowWriter(fps=1/dt))
plt.show()
