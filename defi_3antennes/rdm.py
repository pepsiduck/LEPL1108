import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from position_calc import *

def result_write(positions_x, positions_y, speed_x, speed_y, check_tab):
    f = open("results.txt","w")
    f.write("Position_x Position_y Speed_x Speed_y check\n")
    for i in range(0,len(positions_x)):
        print("SANS UNDERTALE")
        if(positions_x[i] < 0):
            f.write("-")
        else:
            f.write(" ")
        f.write("{:1.7e}".format(abs(positions_x[i])) + "  ")
        if(positions_y[i] < 0):
            f.write("-")
        else:
            f.write(" ")
        f.write("{:1.7e}".format(abs(positions_y[i])) + "  ")
        if(speed_x[i] < 0):
            f.write("-")
        else:
            f.write(" ")
        f.write("{:1.7e}".format(abs(speed_x[i])) + "  ")
        if(speed_y[i] < 0):
            f.write("-")
        else:
            f.write(" ")
        f.write("{:1.7e}".format(abs(speed_y[i])) + "  ")
        if(check_tab[i]):
            f.write("True")
        else:
            f.write("False")
        f.write("\n")
    f.close()

def main():
    # arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str)

    args = parser.parse_args()
    file_name = args.file

    # preprocessing
    data = np.load(file_name)

    Ms = int(data['chirp'][2])
    Mc = int(data['chirp'][3])
    M = int(len(data['data'][0][0]))
    n_frame = int(len(data['data_times']))
    M_pause = int(M/Mc - Ms)
    f0 = data['chirp'][0]
    Ts = data['chirp'][4]
    B = data['chirp'][1]
    Tc = data['chirp'][5]

    print(f"Ms = {Ms}, Mc = {Mc}, M = {M}, M_pause = {M_pause}, n_frame = {n_frame}, f0 = {f0}, Ts = {Ts}, B = {B}, Tc = {Tc}")

    del_index = np.arange(Ms, Ms + M_pause)

    for c in range(1, Mc):
        del_index = np.append(del_index, (del_index[0:M_pause] + c * (M_pause + Ms)))

    signal = np.zeros((3, n_frame, M - Mc * M_pause), dtype=complex)

    for c in range(0, 3):
        for t in range(len(data['data'])):
            e_des = np.array([data['data'][t][c+1] + 0j],dtype=complex)
            sig_t = e_des
            sig_t = np.delete(sig_t, del_index)
            signal[c][t] = sig_t

    signal = np.reshape(signal, (3, n_frame, Mc, Ms))


    k = 5

        #padding constant
    pad_size_Ms = k * Ms
    pad_size_Mc = k* Mc


    signal -= np.mean(signal, axis=1, keepdims=True)   # remove zero velocity
    #signal = np.pad(signal, ((0, 0), (0, (k-1)*Mc), (0, (k - 1) * Ms)), mode='constant', constant_values=0)


    # range doppler map

    rdm = [None, None, None]

    for c in range(0, 3):
        rdm[c] = np.fft.ifft(signal[c], n=pad_size_Ms,axis=2)
        rdm[c] = np.transpose(rdm[c], (0, 2, 1))
        rdm[c] = np.fft.ifft(rdm[c], n=pad_size_Mc,axis=2)
        rdm[c] = np.transpose(rdm[c], (0, 2, 1))
        rdm[c] = np.roll(rdm[c], int(pad_size_Mc / 2)-1, axis=1)
        rdm[c] = np.transpose(rdm[c], (0, 2, 1))


    range_values = [np.arange(0, pad_size_Ms, 1) * (3e8) / (2 * B * k) for i in range(0,3)]
    
    try :
        f = open("calibration.txt","r")
        lines = f.readlines()
        for i in range(0,3):
            range_values[i] += float(lines[i+1][4:18])
            print(float(lines[i+1][4:18]))
        f.close()
    except:
        pass
    
    velocity_values = np.arange(0, (pad_size_Mc) / 2 + 1, 1) * (3e8) / (2 * f0 * pad_size_Mc * Tc)
    velocity_values = np.concatenate((-velocity_values[::-1], velocity_values[1:]))[1:pad_size_Mc + 1]

    
    #r0 = np.zeros(n_frame)
    r1 = np.zeros(n_frame)
    r2 = np.zeros(n_frame)
    r3 = np.zeros(n_frame)

    #u0 = np.zeros(n_frame)
    u1 = np.zeros(n_frame)
    u2 = np.zeros(n_frame)
    u3 = np.zeros(n_frame)

    for frame in range(0,n_frame):
        """
        test = rdm[0][frame][0:(len(rdm[0][frame]) // 2)]
        max_idx0 = np.unravel_index(np.argmax(np.abs(test)), test.shape)
        r0[frame] = max(0,range_values[0][max_idx0[0]])
        u0[frame] = velocity_values[max_idx0[1]]
        """
        test = rdm[0][frame][0:(len(rdm[0][frame]) // 2)]
        max_idx1 = np.unravel_index(np.argmax(np.abs(test)), test.shape)
        r1[frame] = max(0,range_values[0][max_idx1[0]])
        u1[frame] = velocity_values[max_idx1[1]]

        test = rdm[1][frame][0:(len(rdm[1][frame]) // 2)]
        max_idx2 = np.unravel_index(np.argmax(np.abs(test)), test.shape)
        r2[frame] = max(0,range_values[1][max_idx2[0]])
        u2[frame] = velocity_values[max_idx2[1]]

        test = rdm[2][frame][0:(len(rdm[2][frame]) // 2)]
        max_idx3 = np.unravel_index(np.argmax(np.abs(test)), test.shape)
        r3[frame] = max(0,range_values[2][max_idx3[0]])
        u3[frame] = velocity_values[max_idx3[1]]

    try:
        f = open("geometry.txt","r")
        lines = f.readlines()
        e_coord = (float(lines[0][5:19]), float(lines[0][20:34]))
        a_coord = [None, None, None]
        for i in range(0, 3):
            a_coord[i] = (float(lines[i+2][5:19]), float(lines[i+2][20:34]))
        #t_coord = (float(lines[5][5:19]), float(lines[5][20:34])) #uniquement pour la calibration
        f.close()
    except:
        print("Absent or incorrect geometry. bruh")
        exit(1)

    positions = np.array([position_calc([2*r1[frame],2*r2[frame],2*r3[frame]], e_coord, a_coord) for frame in range(0, n_frame)])
    positions_x = np.array([positions[frame][0] for frame in range(0, n_frame)])
    positions_y = np.array([positions[frame][1] for frame in range(0, n_frame)])
    
    speeds = np.array([vitesse_calc([positions_x[frame], positions_y[frame]], e_coord, a_coord, [u1[frame], u2[frame], u3[frame]]) for frame in range(0, n_frame)])
    speeds_x = np.array([speeds[frame][0] for frame in range(0, n_frame)])
    speeds_y = np.array([speeds[frame][1] for frame in range(0, n_frame)])  

    check_tab = [True for frame in range(0,n_frame)]  

    fig, ax = plt.subplots(1, 2)
    for frame in range(0, n_frame):
        if(abs(positions_x[frame]) > 3.8 or positions_y[frame] < 0 or positions_y[frame] > 15):
            ax[0].plot(positions_x[frame], positions_y[frame], 'ro')
            ax[0].quiver(positions_x[frame], positions_y[frame], speeds_x[frame], speeds_y[frame], color='red')
            check_tab[frame] = False
        else:
            ax[0].plot(positions_x[frame], positions_y[frame], 'go')
            ax[0].quiver(positions_x[frame], positions_y[frame], speeds_x[frame], speeds_y[frame], color='green') 

    result_write(positions_x, positions_y, speeds_x, speeds_y, check_tab)
    
    plt.show()
    

if __name__ == '__main__':
    main()
