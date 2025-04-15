import argparse
import numpy as np

def list_of_strings(arg):
    return arg.split(',')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", type=list_of_strings)
    args = parser.parse_args()

    r = np.zeros(4)
    
    for file_name in args.files:
        f = open(file_name.strip(),"r")
        lines = f.readlines()
        for i in range(0, len(r)):
            r[i] += float(lines[i][4:18])
        f.close()

    r /= len(args.files)
    
    f = open("calibration.txt","w")
    for i in range(0, len(r)):
        f.write("a" + str(i) + ": ")
        if(r[i] < 0.0):
            f.write("-")
        else:
            f.write(" ")
        f.write("{:1.7e}".format(abs(r[i])) + "\n")
    
    f.close()

if __name__ == "__main__":
    main()
