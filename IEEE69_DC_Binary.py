import numpy as np
import cmath
import time
# import matplotlib.pyplot as plt
# import pandas as pd

handel_calls = 0
time_start = time.time()


target=0
res=10000
iter=0
plt_y=[]
plt_x=[]
def handel(delt):
    global handel_calls # Increment the counter each time `handel` is called
    handel_calls += 1

    Vslackini=20
    Vslack = 20 + delt
    Pref =[0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
             0, 0, 0, 0, 0, 0, 0, -1, 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    
    Vref =         [Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini,
         Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini,
         Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini,
         Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini,
         Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini,
         Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini,
         Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini, Vslackini]
    Vref = np.array(Vref, dtype=float)
    
    Kpv = [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
             0, 0, 0, 0, 0, 0, 0, 2, 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    Kpv = np.array(Kpv, dtype=float).reshape(-1, 1)
    
    BUS = [
        [Vslack, 0, 1],
        [Vslack, 0, 2],
        [Vslack, 0, 2],
        [Vslack, 0, 2],
        [Vslack, Pref[4] + Vref[4]*Kpv[4], 2], #5
        [Vslack, -2.6/1000, 2],
        [Vslack, -40.4/1000, 2],
        [Vslack, -75/1000, 2],
        [Vslack, -30/1000, 2],
        [Vslack, -28/1000, 2],
        [Vslack, -145/1000, 2],
        [Vslack, -145/1000, 2],
        [Vslack, -8/1000, 2],
        [Vslack, -8/1000, 2],
        [Vslack, Pref[14] + Vref[14]*Kpv[14], 2], #15
        [Vslack, -45.5/1000, 2],
        [Vslack, -60/1000, 2],
        [Vslack, -60/1000, 2],
        [Vslack, 0, 2],
        [Vslack, -1/1000, 2],
        [Vslack, -114/1000, 2],
        [Vslack, -5/1000, 2],
        [Vslack, 0, 2],
        [Vslack, -28/1000, 2],
        [Vslack, 0, 2],
        [Vslack, -14/1000, 2],
        [Vslack, -14/1000, 2],
        [Vslack, -26/1000, 2],
        [Vslack, -26/1000, 2],
        [Vslack, 0, 2],
        [Vslack, 0, 2],
        [Vslack, 0, 2],
        [Vslack, -14/1000, 2],
        [Vslack, -19.5/1000, 2],
        [Vslack, -6/1000, 2],
        [Vslack, -26/1000, 2],
        [Vslack, -26/1000, 2],
        [Vslack, Pref[37] + Vref[37]*Kpv[37], 2], #38
        [Vslack, -24/1000, 2],
        [Vslack, -24/1000, 2],
        [Vslack, -1.2/1000, 2],
        [Vslack, 0, 2],
        [Vslack, -6/1000, 2],
        [Vslack, 0, 2],
        [Vslack, -39.22/1000, 2],
        [Vslack, -39.22/1000, 2],
        [Vslack, 0, 2],
        [Vslack, -79/1000, 2],
        [Vslack, -384.7/1000, 2],
        [Vslack, -384.7/1000, 2],
        [Vslack, -40.5/1000, 2],
        [Vslack, -3.6/1000, 2],
        [Vslack, -4.35/1000, 2],
        [Vslack, -26.4/1000, 2],
        [Vslack, -24.0/1000, 2],
        [Vslack, 0, 2],
        [Vslack, 0, 2],
        [Vslack, 0, 2],
        [Vslack, -100/1000, 2],
        [Vslack, 0, 2],
        [Vslack, -1244/1000, 2],
        [Vslack, -32/1000, 2],
        [Vslack, 0, 2],
        [Vslack, -227/1000, 2],
        [Vslack, -59/1000, 2],
        [Vslack, -18/1000, 2],
        [Vslack, -18/1000, 2],
        [Vslack, -28/1000, 2],
        [Vslack, -28/1000, 2]
    ]
    
    
    BUS = np.array(BUS, dtype=object)
    
    
    VB=20
    SB=20
    BUS[:,0:1]=BUS[:,0:1]/VB
    BUS[:,1:2]=BUS[:,1:2]/SB
    ZB=VB*VB/SB
    # Kpv = Kpv * VB / SB
    
    Topu = [
        [1, 2, 0.0005],
        [2, 3, 0.0005],
        [3, 4, 0.0015],
        [4, 5, 0.0251],
        [5, 6, 0.3660],
        [6, 7, 0.3811],
        [7, 8, 0.0922],
        [8, 9, 0.0493],
        [9, 10, 0.8190],
        [10, 11, 0.1872],
        [11, 12, 0.7114],
        [12, 13, 1.03],
        [13, 14, 1.044],
        [14, 15, 1.058],
        [15, 16, 0.1966],
        [16, 17, 0.3744],
        [17, 18, 0.0047],
        [18, 19, 0.3276],
        [19, 20, 0.2106],
        [20, 21, 0.3416],
        [21, 22, 0.014],
        [22, 23, 0.1591],
        [23, 24, 0.3463],
        [24, 25, 0.7488],
        [25, 26, 0.3089],
        [26, 27, 0.1732],
        [3, 28, 0.0044],
        [28, 29, 0.0640],
        [29, 30, 0.3978],
        [30, 31, 0.0702],
        [31, 32, 0.351],
        [32, 33, 0.839],
        [33, 34, 1.708],
        [34, 35, 1.474],
        [3, 36, 0.0044],
        [36, 37, 0.0640],
        [37, 38, 0.1053],
        [38, 39, 0.0304],
        [39, 40, 0.0018],
        [40, 41, 0.7283],
        [41, 42, 0.3100],
        [42, 43, 0.0410],
        [43, 44, 0.0092],
        [44, 45, 0.1089],
        [45, 46, 0.0009],
        [4, 47, 0.0034],
        [47, 48, 0.0851],
        [48, 49, 0.2898],
        [49, 50, 0.0822],
        [8, 51, 0.0928],
        [51, 52, 0.3319],
        [9, 53, 0.1740],
        [53, 54, 0.2030],
        [54, 55, 0.2842],
        [55, 56, 0.2813],
        [56, 57, 1.5900],
        [57, 58, 0.7837],
        [58, 59, 0.3042],
        [59, 60, 0.3861],
        [60, 61, 0.5075],
        [61, 62, 0.0974],
        [62, 63, 0.1450],
        [63, 64, 0.7105],
        [64, 65, 1.0410],
        [11, 66, 0.2012],
        [66, 67, 0.0047],
        [12, 68, 0.7394],
        [68, 69, 0.0047]
    ]
    
    Topu=np.array(Topu)
    
    
    #print(Topu[:,2:3])
    Topu[:,2:3]=Topu[:,2:3]/ZB
    Topu[:,2:3]=1/Topu[:,2:3]
    #print(Topu[:,2:3])
    nbus=69
    V=BUS[:,0:1]
    type=BUS[:,2:3]
    
    pq=np.argwhere(type[:,0]==2)
    
    npv=(type==1).sum()
    npq=(type==2).sum()
    G =np.zeros((69,69))
    for k in range(Topu.shape[0]):
        t1=Topu[k][0]
        t1=int(t1)
        t2=Topu[k][1]
        t2=int(t2)
        g=Topu[k][2]
        G[t1-1][t1-1]=G[t1-1][t1-1]+g
        G[t1-1][t2-1]=G[t1-1][t2-1]-g
        G[t2-1][t1-1]=G[t2-1][t1-1]-g
        G[t2-1][t2-1]=G[t2-1][t2-1]+g
    #print(G)
    Pini=BUS[:,1:2]
    Iteration=0
    Tol=1
    
    
    while(Tol>0.00001 and Iteration<=30):
        P=np.zeros((nbus,1))
        for i in range(0,nbus):
            for j in range(0,nbus):
                P[i]=P[i]+V[i]*V[j]*G[i][j]
        dPa=Pini - P - V * Kpv
        dP=dPa[1:69]
        N=np.zeros((nbus-1,npq))
        for i in range(0,nbus-1):
            m=i+1
            for k in range(0,npq):
                n=pq[k] 
                if n==m:
                    N[i][k]=(P[m]+V[m]*V[m]*G[m][m])/V[m] + Kpv[m]
                else:
                    N[i][k]=V[m]*G[m][n]
        # print(N)
        dV=np.linalg.inv(N).dot(dP)
        V[1:,:]=dV+V[1:,:]
        Iteration=Iteration+1
        Tol=max(abs(dP))
    V=V*VB
    # print(V)
    # print("xxxxxxxxxxxxxxxxx")
    #print(dP)
    Ga =np.zeros((69,69))
    for k in range(Topu.shape[0]):
        t1=Topu[k][0]
        t1=int(t1)
        t2=Topu[k][1]
        t2=int(t2)
        g=Topu[k][2]/ZB
        Ga[t1-1][t1-1]=Ga[t1-1][t1-1]+g
        Ga[t1-1][t2-1]=Ga[t1-1][t2-1]-g
        Ga[t2-1][t1-1]=Ga[t2-1][t1-1]-g
        Ga[t2-1][t2-1]=Ga[t2-1][t2-1]+g
    P_G1=0
    for j in range(0,69):
        P_G1 = P_G1 + V[0] * V[j] * Ga[0][j]
    # P=np.hstack((P_G1,BUS[:,1]*SB))
    # print(P*20)
# offset_real=target-P[1]/V[1]
    offset=abs(target-20*( (Vref[0]-V[0])*Kpv[0]+(Pref[0]-P[0]) ))
    # plt_x.append(delt)
    # plt_y.append(offset_real)
    flag=0
    if offset<0.001:
        #res=offset
        # print("the final it index: "+str(iter))
        # print("init power on Bus2: "+str(Vslack))
        #I2=P[1]/V[1]
        flag=1
        print(f"F = {offset}")
    return offset,flag

def find_Peak():
    Left_index = -2
    Right_index = 2
    step = 0.00001
    flag_all = 0
    max_iterations = 30  # maximum allowable search times
    count = 0

    Left_offset, flag_left = handel(Left_index)
    Right_offset, flag_right = handel(Right_index)
    flag_all = max(flag_left, flag_right)

    while flag_all != 1 and count < max_iterations:
        handel_calls_in_iteration = 0  # Counter to track `handel` calls per iteration

        mid_index = Left_index + ((Right_index - Left_index) / 2)
        mid_offset, mid_flag = handel(mid_index)
        handel_calls_in_iteration += 1  # Increment the counter

        if mid_flag == 1:
            flag_all = 1
            # print(f"Target found at voltage deviation = {mid_index} after {count + 1} searches.")
            return mid_index, count + 1

        mid_offset_min_step, mid_offset_min_flag = handel(mid_index - step)
        handel_calls_in_iteration += 1  # Increment the counter

        mid_offset_add_step, mid_offset_add_flag = handel(mid_index + step)
        handel_calls_in_iteration += 1  # Increment the counter


        if mid_offset < mid_offset_add_step:
            Right_index = mid_index
        else:
            Left_index = mid_index

        count += 1

        print(f"Iteration {count}: handel was called {handel_calls_in_iteration} times.")

    # if reach to the maximum
    if count >= max_iterations:
        print(f"Search terminated after {max_iterations} iterations without finding the target.")
        return None, count

peak_result = find_Peak()
print(f"Peak found at: {peak_result}")

time_end = time.time()
time_sum = time_end - time_start
print(f"'handel' function was called {handel_calls} times.")


print(time_sum)