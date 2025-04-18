import math
import random
import numpy as np

def vitesse_calc(cible, e_coord, a_coord, speeds):
    A = np.zeros((len(speeds), 2))
    B = np.zeros(len(speeds))
    vector_e_t = [cible[0]-e_coord[0], cible[1]-e_coord[1]]
    vector_e_t = vector_e_t / np.linalg.norm(vector_e_t)
    for i in range(0,len(speeds)):
        B[i] = 2 * speeds[i]
        vector_a_t = [cible[0]-a_coord[i][0], cible[1]-a_coord[i][1]]
        vector_a_t = vector_a_t / np.linalg.norm(vector_a_t)
        A[i][0] = vector_e_t[0] + vector_a_t[0]
        A[i][1] = vector_e_t[1] + vector_a_t[1]

    #print(speeds)

    x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    return (x[0],x[1])

alpha = 0.05

"""
    distances : tableau contenant les distances des rdm
    e_coord : tuple des coordonées de l'émetteur
    a_coord : tableau de tuples contenant les coordonées des antennes
"""
def position_calc(distances, e_coord, a_coord):
 
    #print(e_coord)
    #print(a_coord)

    ret_x = 0.0
    ret_y = 0.0
    func_min = 1e10

    #data = [None for j in range(0, 20)]

    for j in range(0, 100):
        x = random.uniform(-20.0, 20.0)
        y = random.uniform(-20.0, 20.0)

        for c in range(0, 100):
            grad_x = 0.0
            grad_y = 0.0

            #gradient calculation
            de = math.sqrt((x - e_coord[0])**2 + (y - e_coord[1])**2)
            for i in range(0,len(a_coord)):
                di = math.sqrt((x - a_coord[i][0])**2 + (y - a_coord[i][1])**2)  
                f = distances[i] - de - di
                delta_x = -1.0 * (((x - e_coord[0]) / de) + ((x - a_coord[i][0]) / di))   
                delta_y = -1.0 * (((y - e_coord[1]) / de) + ((y - a_coord[i][1]) / di))  
                grad_x += 2 * f * delta_x 
                grad_y += 2 * f * delta_y

            if(grad_x == 0.0 and grad_y == 0.0):
                break
            #gradient application
            x -= grad_x * alpha
            y -= grad_y * alpha

        func = 0.0
        for i in range(0,len(a_coord)):
            func += (distances[i] - math.sqrt((x - e_coord[0])**2 + (y - e_coord[1])**2) - math.sqrt((x - a_coord[i][0])**2 + (y - a_coord[i][1])**2))**2
        
        #data[j] = np.array([x, y, func])

        if(func < func_min) :
            ret_x = x
            ret_y = y
            func_min = func

    """
    for j in range(0, 20):
        print(str(data[j][0]) + "   " + str(data[j][1]) + "   " + str(data[j][2]))
    print("\n\n")
    """
    #print(str(np.array(distances)) + "||(" + str(ret_x) + " ; " + str(ret_y) + ")||" + str(func_min))
    """
    ax_x = np.linspace(-3.2, 5, num = 25)
    ax_y = np.linspace(0, 5, num=25)
    D = np.zeros((len(ax_x), len(ax_y)))
    
    for x in range(0, len(ax_x)):
        for y in range(0, len(ax_y)):
            D[x][y] = 0.0
            de = math.sqrt((ax_x[x] - e_coord[0])**2 + (ax_y[y] - e_coord[1])**2)
            for i in range(0, len(a_coord)) :
                D[x][y] += (distances[i] - de - math.sqrt((ax_x[x] - a_coord[i][0])**2 + (ax_y[y] - a_coord[i][1])**2))**2

    min_index = np.unravel_index(np.argmin(D), D.shape)
    """
    #return (ax_x[min_index[0]], ax_y[min_index[1]])
    return (ret_x, ret_y)


if __name__ == "__main__":
    distances = [2*4, 2*3.67, 2*4.3, 2*3.99]
    e_coord = (0.0, 0.0)
    a_coord = [(0.0, 0.0), (-3.2, 3.0), (-2.3, 0.0), (-3.1, 1.5)]

    print(position_calc(distances,e_coord,a_coord))

