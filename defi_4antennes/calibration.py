import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

def main():
    # arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str)
    parser.add_argument("-n", "--number", type=int)

    args = parser.parse_args()
    file_name = args.file
    number = args.number
    if(file_name == None or number == None):
        print("Missing arguments.\n")
        exit(1)

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

    signal = np.zeros((4, n_frame, M - Mc * M_pause), dtype=complex)

    for c in range(0, 4):
        for t in range(len(data['data'])):
            e_des = np.array([data['data'][t][c] + 0j],dtype=complex)
            sig_t = e_des
            sig_t = np.delete(sig_t, del_index)
            signal[c][t] = sig_t

    signal = np.reshape(signal, (4, n_frame, Mc, Ms))


    k = 5

        #padding constant
    pad_size_Ms = k * Ms
    pad_size_Mc = k* Mc


    signal -= np.mean(signal, axis=1, keepdims=True)   # remove zero velocity
    #signal = np.pad(signal, ((0, 0), (0, (k-1)*Mc), (0, (k - 1) * Ms)), mode='constant', constant_values=0)


    # range doppler map

    rdm = [None, None, None, None]

    for c in range(0, 4):
        rdm[c] = np.fft.ifft(signal[c], n=pad_size_Ms,axis=2)
        rdm[c] = np.transpose(rdm[c], (0, 2, 1))
        rdm[c] = np.fft.ifft(rdm[c], n=pad_size_Mc,axis=2)
        rdm[c] = np.transpose(rdm[c], (0, 2, 1))
        rdm[c] = np.roll(rdm[c], int(pad_size_Mc / 2)-1, axis=1)
        rdm[c] = np.transpose(rdm[c], (0, 2, 1))


    range_values = np.arange(0, pad_size_Ms, 1) * (3e8) / (2 * B * k)
    velocity_values = np.arange(0, (pad_size_Mc) / 2 + 1, 1) * (3e8) / (2 * f0 * pad_size_Mc * Tc)
    velocity_values = np.concatenate((-velocity_values[::-1], velocity_values[1:]))[1:pad_size_Mc + 1]

    r0 = 0.0
    r1 = 0.0
    r2 = 0.0
    r3 = 0.0

    for frame in range(0,n_frame):

        test = rdm[0][frame][0:(len(rdm[0][frame]) // 2)]
        max_idx0 = np.unravel_index(np.argmax(np.abs(test)), test.shape)
        r0 += (range_values[max_idx0[0]])

        test = rdm[1][frame][0:(len(rdm[1][frame]) // 2)]
        max_idx1 = np.unravel_index(np.argmax(np.abs(test)), test.shape)
        r1 += (range_values[max_idx1[0]])

        test = rdm[2][frame][0:(len(rdm[2][frame]) // 2)]
        max_idx2 = np.unravel_index(np.argmax(np.abs(test)), test.shape)
        r2 += (range_values[max_idx2[0]])

        test = rdm[3][frame][0:(len(rdm[3][frame]) // 2)]
        max_idx3 = np.unravel_index(np.argmax(np.abs(test)), test.shape)
        r3 += (range_values[max_idx3[0]])

    r0 /= n_frame
    r1 /= n_frame
    r2 /= n_frame
    r3 /= n_frame
    r = [r0, r1, r2, r3]
    
    try :
        f = open("geometry.txt", "r")
        lines = f.readlines()

        e_coord = (float(lines[0][5:19]), float(lines[0][20:34]))
        a_coord = [None, None, None, None]
        for i in range(0, 4):
            a_coord[i] = (float(lines[i+1][5:19]), float(lines[i+1][20:34]))
        t_coord = (float(lines[5][5:19]), float(lines[5][20:34]))#uniquement pour la calibration
        f.close() 
    except:
        print("Absent or incorrect geometry. bruh")
        exit(1)

    de = math.sqrt((e_coord[0] - t_coord[0])**2 + (e_coord[1] - t_coord[1])**2)
    d = [((de + math.sqrt((a_coord[i][0] - t_coord[0])**2 + (a_coord[i][1] - t_coord[1])**2)) / 2.0) for i in range(0,4)]

    f = open("calibration" + str(number) + ".txt", "w")
    for i in range(0, 4):
        to_write = d[i] - r[i]
        print(str(d[i]) + " - " + str(r[i]) + " = " + str(to_write))
        f.write("a" + str(i) + ": ")
        if(to_write < 0):
            f.write("-")
        else:
            f.write(" ")
        f.write("{:1.7e}".format(abs(to_write)) + "\n")
    f.close()

if __name__ == '__main__':
    main()
