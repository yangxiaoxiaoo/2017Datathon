"""regression 2 - XiaoyueGong"""

import time
import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def get_means_deviations(numbers):
    total = 0
    devia_sum = 0
    n = len(numbers)
    for i in range(n):
        total += int(numbers[i])
    ave = float(total)/n
    for i in range(n):
        devia_sum += math.pow(int(numbers[i])-ave, 2)
    devia = float(devia_sum)/n
    return ave, devia

def write_normalized(fname,nname):
    norm_file = open(nname, 'w')
    with open(fname, 'r') as f:
        units_text = f.readlines()
        unit_str = [line.split(',') for line in units_text]
    units = []
    for unit in unit_str:
        house = []
        for ka in unit:
            fea = float(ka)
            #print(fea, "fea")
            house.append(fea)
        units.append(house)
    print(units)
    numbers_list = []
    aves = []
    devias = []
    for i in range(len(units[0])):
        numbers = [units[j][i] for j in range(len(units))]
        numbers_list.append(numbers)
        ave, devia = get_means_deviations(numbers)
        aves.append(ave)
        devias.append(devia)
        print(ave, devia, "ave, devia", i)
    for j in range(len(units)):
        for i in range(len(units[0])):
            a = float(units[j][i])-aves[i]
            b = devias[i]
            nf = float(a)/b
            #print(f)
            norm_file.write(str(nf))
            norm_file.write(',')
        norm_file.write("\n")
    norm_file.close()
    f.close()
    return aves, devias, numbers_list


def find_minimizing_features(fname,alpha, max_it = 200):
    write_normalized(fname,"normalized.txt")
    with open("normalized.txt", 'r') as fi:
        units_text = fi.readlines()           
        unit_str = [line.split(',') for line in units_text]
    units = []
    for unit in unit_str:
        house = []
        for k in range(len(unit)-1):
            a = unit[k].strip()
            print(a)
            fea = float(a)
            #print(fea, "fea")
            house.append(fea)
        units.append(house)
    print(units,"units again")
    m = len(units)
    w = [0,0,0]
    for i in range(m):
        units[i].insert(0,1)
    print(units, "printing normalized units")
    while max_it:
        w_new = []
        Jw = 0
        for j in range(len(w)):
            prod = 0
            for i in range(m):
                diff = w[0]+w[1]*units[i][1]+w[2]*units[i][2]-units[i][3]
                prod += diff*units[i][j]
            w_new.append(w[j] - alpha*prod/float(m))
        w = w_new
        max_it -= 1
        for i in range(m):
            diff = w[0]+w[1]*float(units[i][1])+w[2]*float(units[i][2])-float(units[i][3])
            Jw += (diff*diff/float(2*m))
        print(Jw, "Jw")
    fi.close()
    return w, Jw

def limit_minimize(fname,alpha, N):
    ws = []
    Jws = []
    for val in N:
        print("limit to", val, "iterations")
        w, Jw = find_minimizing_features(fname, alpha, val)
        ws.append(w)
        Jws.append(Jw)
    return ws, Jws

def predict(fname, nname, area, bedrms,alpha):
    aves, devias, l = write_normalized(fname, nname)
    narea = (area-aves[0])/float(devias[0])
    nbedrms = (bedrms-aves[1])/float(devias[1])
    w, Jw = find_minimizing_features(fname,alpha)
    func = w[0] + w[1]*narea + w[2]*nbedrms
    print(func,"func")
    price = func*devias[2]+aves[2]
    print(price, "price prediction")
    return price

def stochastic_grad(fname, alpha, max_it):
    write_normalized(fname,"normalized.txt")
    with open("normalized.txt", 'r') as fi:
        units_text = fi.readlines()           
        unit_str = [line.split(',') for line in units_text]
    units = []
    for unit in unit_str:
        house = []
        for k in range(len(unit)-1):
            a = unit[k].strip()
            print(a)
            fea = float(a)
            #print(fea, "fea")
            house.append(fea)
        units.append(house)
        #print(units, "units")
    m = len(units)
    for i in range(m):
        units[i].insert(0,1)
    w = [0,0,0]
    while max_it > 0:
        Jw = 0
        max_it -= 1
        for i in range(m):
            w_new = []
            for j in range(len(w)):
                diff = w[0]+w[1]*float(units[i][1])+w[2]*float(units[i][2])-float(units[i][3])
                prod = alpha*diff*float(units[i][j])
                w_new.append(w[j] - prod)
            w = w_new
            print(w, "w of k")
        for i in range(m):
            diff = w[0]+w[1]*float(units[i][1])+w[2]*float(units[i][2])-float(units[i][3])
            Jw += diff*diff/float(2*m)
        print(Jw, "Sto pass", (3-max_it))
    #print(w, "w")
    fi.close()
    return w, Jw

def main():
    x = [1,3,5]
    #print(find_minimizing_features("housing.txt", 0.5, 80),"alpha =0.5")
    #print(find_minimizing_features("housing.txt", 0.05,80), "alpha =0.05")
    
    #print(find_minimizing_features("housing.txt", 1.5,80), "alpha = 1.5")

    #print(get_means_deviations(x))
    #print(write_normalized("housing.txt", "normalized.txt"))
    find_minimizing_features("housing.txt",0.3)
    predict("housing.txt","normalized.txt",1650,3,0.1)
    print("sto", stochastic_grad("housing.txt",0.01,3))
    print("vanilla", find_minimizing_features("housing.txt",0.01, 80))
    print("\nPLOTS")
    #datapoints
    #Alphas = [0.01, 0.1, 0.3]
    N = [10, 20, 30, 40, 50, 60,70, 80]
    N_w1, N_J1 = limit_minimize("housing.txt", 0.01, N)
    plt.figure(1, figsize=(8, 5))
    plt.title("Jw vs Iterations for alpha = 0.01")
    plt.plot(N, N_J1, 'bo-', linewidth=1)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Jw")
    plt.savefig('Jw0.01.pdf')
    #plot number of passes
    N = [10, 20, 30, 40, 50, 60,70, 80]
    N_w, N_J2 = limit_minimize("housing.txt", 0.1, N)
    plt.figure(2, figsize=(8, 5))
    plt.title("Jw vs Iterations for alpha = 0.1")
    plt.plot(N, N_J2, 'bo-', linewidth=1)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Jw")
    plt.savefig('Jw2.pdf')

    N = [10, 20, 30, 40, 50, 60,70, 80]
    N_w, N_J3 = limit_minimize("housing.txt", 0.3, N)
    plt.figure(3, figsize=(8, 5))
    plt.title("Jw vs Iterations for alpha = 0.3")
    plt.plot(N, N_J3, 'bo-', linewidth=1)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Jw")
    plt.savefig('Jw3.pdf')

    #show plot
    print("Displaying plots...")
    plt.show()





if __name__ == '__main__':
    try:
        start = time.time()
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("---completed in %ss---" % str(time.time() - start))
