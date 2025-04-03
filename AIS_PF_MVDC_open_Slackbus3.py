"""
@author: Pingyang Sun UNSW Sydney

"""

import numpy as np
import cmath
import time
# import matplotlib.pyplot as plt
# import pandas as pd

handel_calls = 0
time_start = time.time()

# Create list to save data
# log_data = {
#     'iteration': [],
#     'delt': [],
#     'offset': [],
# }

target=0
res=10000
iter=0
plt_y=[]
plt_x=[]
def handel(delt):
    global handel_calls # Increment the counter each time `handel` is called
    handel_calls += 1

    # Droop constants [I/V control; Pcontrol; Pcontrol; Icontrol; P/Vcontrol; P/Vcontrol]
    K_d = np.array([0.1, 0, 0, 0, 3, 2.5]).reshape(-1, 1)  # K_d = 0 means constant power or current control

    V1ref = 20
    V1_initial = 20

    V2ref = 20
    V3ref = 20
    V4ref = 20
    V5ref = 20
    V6ref = 20

    P5ref = 6
    P6ref = 5
    I1ref = 0.2
    P2ref = -5
    P3ref = 0
    I4ref = -0.15

    Vslack = V1_initial + delt;  # obtain initial value

    # Bus type [V; P; I; P/V]
    BUS = [[Vslack, 0, 0, 0, 0, 1], 
           [Vslack, P2ref, 0, 0, 0, 2], 
           [Vslack, P3ref, 0, 0, 0, 2],
           [Vslack, 0, I4ref, 0, 0, 3], 
           [Vslack, 0, 0, P5ref+ V5ref * K_d[4], 0, 4],
           [Vslack, 0, 0, P6ref+ V6ref * K_d[5], 0, 4]]

    BUS = np.array(BUS, dtype=object)


    # Voltage and power base value
    VB = 20
    SB = 20
    ZB = VB * VB / SB  # Base impedance

    # p.u. value conversion
    BUS[:, 0:1] = BUS[:, 0:1] / VB
    BUS[:, 1:2] = BUS[:, 1:2] / SB
    BUS[:, 2:3] = BUS[:, 2:3] / (SB / VB)
    BUS[:, 3:4] = BUS[:, 3:4] / SB
    BUS[:, 4:5] = BUS[:, 4:5] / (SB / VB)

    K_d[4:6] = K_d[4:6] * VB / SB  # pu conversion for P/V droop constants MW/kV
    # K_d[0] = K_d[0] * VB / (SB / VB)  # pu conversion for I/V droop constants kA/kV

    # line parameters
    Line = [[1, 4, 1.2], [3, 6, 1.0], [2, 5, 1.2], [2, 3, 0.8], [3, 4, 0.8]]  # line1. line2, line3, line4, line5
    Line = np.array(Line)

    # print(Line[:,2:3])
    Line[:, 2:3] = Line[:, 2:3] / ZB
    Line[:, 2:3] = 1 / Line[:, 2:3]
    # print(Line[:,2:3])

    # number of buses
    nbus = 6

    # rated voltage
    V = BUS[:, 0:1]

    # bus type location
    type = BUS[:, 5:6]

    # bus type classification
    v_b = np.argwhere(type[:, 0] == 1)
    p_b = np.argwhere(type[:, 0] == 2)
    i_b = np.argwhere(type[:, 0] == 3)
    pv_b = np.argwhere(type[:, 0] == 4)
    # iv_b = np.argwhere(type[:, 0] == 5)

    # number of buses in each bus type
    nv_b = (type == 1).sum()
    np_b = (type == 2).sum()
    ni_b = (type == 3).sum()
    npv_b = (type == 4).sum()
    # niv_b = (type == 5).sum()

    # line admittance matrix
    G = np.zeros((6, 6))
    for k in range(Line.shape[0]):  # number of lines
        t1 = Line[k][0]
        t1 = int(t1)
        t2 = Line[k][1]
        t2 = int(t2)
        g = Line[k][2]
        G[t1 - 1][t1 - 1] = G[t1 - 1][t1 - 1] + g
        G[t1 - 1][t2 - 1] = G[t1 - 1][t2 - 1] - g
        G[t2 - 1][t1 - 1] = G[t2 - 1][t1 - 1] - g
        G[t2 - 1][t2 - 1] = G[t2 - 1][t2 - 1] + g
    # print(G)

    # initial values in each bus
    PVini = BUS[:, 3:4]
    # IVini = BUS[:, 4:5]
    Pini = BUS[:, 1:2]
    Iini = BUS[:, 2:3]

    # NR iteration start
    Iteration = 0
    Tol = 1  # tolerance

    while (Tol > 0.00001 and Iteration <= 30):
        P = np.zeros((nbus, 1))
        I = np.zeros((nbus, 1))

        for i in range(0, nbus):
            for j in range(0, nbus):
                P[i] = P[i] + V[i] * V[j] * G[i][j]
                I[i] = I[i] + V[j] * G[i][j]

        dPV = PVini[4:6] - (V[4:6] * K_d[4:6] + P[4:6])
        # dIV = IVini[2:3] - (V[2:3] * K_d[2:3] + I[2:3])
        dP = Pini[1:3] - P[1:3]  # delta
        dI = Iini[3:4] - I[3:4]


        # Sub-Jacobian matrices
        NPP = np.zeros((np_b, np_b))
        NPI = np.zeros((np_b, ni_b))
        NPPV = np.zeros((np_b, npv_b))
        # NPIV = np.zeros((np_b, niv_b))

        NIP = np.zeros((ni_b, np_b))
        NII = np.zeros((ni_b, ni_b))
        NIPV = np.zeros((ni_b, npv_b))
        # NIIV = np.zeros((ni_b, niv_b))

        NPVP = np.zeros((npv_b, np_b))
        NPVI = np.zeros((npv_b, ni_b))
        NPVPV = np.zeros((npv_b, npv_b))
        # NPVIV = np.zeros((npv_b, niv_b))

        # NIVP = np.zeros((niv_b, np_b))
        # NIVI = np.zeros((niv_b, ni_b))
        # NIVPV = np.zeros((niv_b, npv_b))
        # NIVIV = np.zeros((niv_b, niv_b))

        #################### NPP ########################## diagonal
        for i in range(0, np_b):
            m = p_b[i][0]
            for k in range(0, np_b):
                n = p_b[k][0]
                if n == m:
                    NPP[i][k] = (P[m] + V[m] * V[m] * G[m][m]) / V[m]
                else:
                    NPP[i][k] = V[m] * G[m][n]
        #################### NPI ########################## non-diagonal
        for i in range(0, np_b):
            m = p_b[i][0]
            for k in range(0, ni_b):
                n = i_b[k][0]
        NPI[i][k] = V[m] * G[m][n]
        #################### NPPV ########################## non-diagonal
        for i in range(0, np_b):
            m = p_b[i][0]
            for k in range(0, npv_b):
                n = pv_b[k][0]
        NPPV[i][k] = V[m] * G[m][n]
        #################### NPIV ########################## non-diagonal
        # for i in range(0, np_b):
        #     m = p_b[i][0]
        #     for k in range(0, niv_b):
        #         n = iv_b[k][0]
        # NPIV[i][k] = V[m] * G[m][n]

        #################### NIP ########################## non-diagonal
        for i in range(0, ni_b):
            m = i_b[i][0]
            for k in range(0, np_b):
                n = p_b[k][0]
        NIP[i][k] = G[m][n]
        #################### NII ########################## diagonal
        for i in range(0, ni_b):
            m = i_b[i][0]
            for k in range(0, ni_b):
                n = i_b[k][0]
                if n == m:
                    NII[i][k] = G[m][m]
                else:
                    NII[i][k] = G[m][n]
        #################### NIPV ########################## non-diagonal
        for i in range(0, ni_b):
            m = i_b[i][0]
            for k in range(0, npv_b):
                n = pv_b[k][0]
        NIPV[i][k] = G[m][n]
        #################### NIIV ########################## non-diagonal
        # for i in range(0, ni_b):
        #     m = i_b[i][0]
        #     for k in range(0, niv_b):
        #         n = iv_b[k][0]
        # NIIV[i][k] = G[m][n]

        #################### NPVP ########################## non-diagonal
        for i in range(0, npv_b):
            m = pv_b[i][0]
            for k in range(0, np_b):
                n = p_b[k][0]
        NPVP[i][k] = V[m] * G[m][n]
        #################### NPVI ########################## non-diagonal
        for i in range(0, npv_b):
            m = pv_b[i][0]
            for k in range(0, ni_b):
                n = i_b[k][0]
        NPVI[i][k] = V[m] * G[m][n]
        #################### NPVPV ########################## diagonal
        for i in range(0, npv_b):
            m = pv_b[i][0]
            for k in range(0, npv_b):
                n = pv_b[k][0]
                if n == m:
                    NPVPV[i][k] = (P[m] + V[m] * V[m] * G[m][m]) / V[m] + K_d[m]  # K_d is only in the diagonal elements
                else:
                    NPVPV[i][k] = V[m] * G[m][n]
        #################### NPVIV ########################## non-diagonal
        # for i in range(0, npv_b):
        #     m = pv_b[i][0]
        #     for k in range(0, niv_b):
        #         n = iv_b[k][0]
        # NPVIV[i][k] = V[m] * G[m][n]

        #################### NIVP ########################## non-diagonal
        # for i in range(0, niv_b):
        #     m = iv_b[i][0]
        #     for k in range(0, np_b):
        #         n = p_b[k][0]
        # NIVP[i][k] = G[m][n]
        #################### NIVI ########################## non-diagonal
        # for i in range(0, niv_b):
        #     m = iv_b[i][0]
        #     for k in range(0, ni_b):
        #         n = i_b[k][0]
        # NIVI[i][k] = G[m][n]
        #################### NIVPV ########################## non-diagonal
        # for i in range(0, niv_b):
        #     m = iv_b[i][0]
        #     for k in range(0, npv_b):
        #         n = pv_b[k][0]
        # NIVPV[i][k] = G[m][n]
        #################### NIVIV ########################## diagonal
        # for i in range(0, niv_b):
        #     m = iv_b[i][0]
        #     for k in range(0, niv_b):
        #         n = iv_b[k][0]
        #         if n == m:
        #             NIVIV[i][k] = G[m][m] + K_d[m]  # K_d is only in the diagonal elements
        #         else:
        #             NIVIV[i][k] = G[m][n]

        # Nh1 = np.hstack((NPP, NPI, NPPV, NPIV))
        # Nh2 = np.hstack((NIP, NII, NIPV, NIIV))
        # Nh3 = np.hstack((NPVP, NPVI, NPVPV, NPVIV))
        # Nh4 = np.hstack((NIVP, NIVI, NIVPV, NIVIV))
        Nh1 = np.hstack((NPP, NPI, NPPV))
        Nh2 = np.hstack((NIP, NII, NIPV))
        Nh3 = np.hstack((NPVP, NPVI, NPVPV))
        # Nh4 = np.hstack((NIVP, NIVI, NIVPV, NIVIV))

        J = np.vstack((Nh1, Nh2, Nh3))
        dfunction = np.vstack((dP, dI, dPV))
        # print(dfunction)

        # print(N)
        dV = np.linalg.inv(J).dot(dfunction)
        V[1:, :] = dV + V[1:, :]
        Iteration = Iteration + 1
        Tol = max(
    max(abs(x) for x in dP), 
    max(abs(x) for x in dI),
    max(abs(x) for x in dPV)
)
    V = V * VB
    # print(V)

    Ga = np.zeros((6, 6))
    for k in range(Line.shape[0]):
        t1 = Line[k][0]
        t1 = int(t1)
        t2 = Line[k][1]
        t2 = int(t2)
        g = Line[k][2] / ZB
        Ga[t1 - 1][t1 - 1] = Ga[t1 - 1][t1 - 1] + g
        Ga[t1 - 1][t2 - 1] = Ga[t1 - 1][t2 - 1] - g
        Ga[t2 - 1][t1 - 1] = Ga[t2 - 1][t1 - 1] - g
        Ga[t2 - 1][t2 - 1] = Ga[t2 - 1][t2 - 1] + g
    P_B1 = 0
    for j in range(0, 6):
        P_B1 = P_B1 + V[0] * V[j] * Ga[0][j]  # P control

    P_B5 = BUS[4, 3] * SB - V[4] * K_d[4] / (VB / SB)  # P/V control
    P_B6 = BUS[5, 3] * SB - V[5] * K_d[5] / (VB / SB)  # P/V control
    P_B2 = BUS[1, 1] * SB  # P control
    P_B3 = BUS[2, 1] * SB  # P control
    P_B4 = BUS[3, 2] * (SB / VB) * V[3]  # I control


    P = np.hstack((P_B1, P_B2, P_B3, P_B4, P_B5, P_B6))


    # print(P)
    # offset_real=target - ( (V1ref-V[0])*K_d[0]+(P1ref-P[0]) )
    offset=abs(target-( (V1ref-V[0])*K_d[0]+(I1ref-I[0]) ))
    # plt_x.append(delt)
    # plt_y.append(offset_real)

    # Document data
    # log_data['iteration'].append(handel_calls)
    # log_data['delt'].append(float(delt))
    # log_data['offset'].append(float(offset)) 


    # print(f"delt: {delt}")
    # print(f"offset: {offset}")


    flag=0
    if offset<0.001:
        #res=offset
        # print("the final it index: "+str(iter))
        # print("init power on Bus2: "+str(Vslack))
        #I2=P[1]/V[1]
        flag=1
        # print(f"F = {offset}")
    return offset,flag

def find_Peak():
    Left_index=-2
    Right_index=2
    step=0.00001
    flag_all=0
    max_iterations = 100  # maximum allowable search times=
    Left_offset,flag_left=handel(Left_index)
    Right_offset,flag_right=handel(Right_index)
    flag_all=max(flag_left,flag_right)
    count=0
    
    while flag_all != 1 and count < max_iterations:
        handel_calls_in_iteration = 0  # Counter to track `handel` calls per iteration
        mid_calls = 0  # Counter for handel(mid_index)
        vertex_calls = 0  # Counter for handel(vertex_index)

        mid_index = Left_index + ((Right_index - Left_index) / 2)
        mid_offset,mid_flag=handel(mid_index)
        handel_calls_in_iteration += 1  # Increment the counter
        mid_calls += 1  # Increment mid_calls counter
        if mid_flag==1:
            flag_all=1
            return mid_index,count+1

        # Initial three points for Lagrange interpolation
        x0, y0 = Left_index, Left_offset
        x1, y1 = mid_index, mid_offset
        x2, y2 = Right_index, Right_offset

        # Coefficients determination for Lagrange interpolation
        denom = (x0-x1)*(x0-x2)*(x1-x2)
        A = (x2 * (y1-y0) + x1 * (y0-y2) + x0 * (y2-y1)) / denom
        B = (x2**2 * (y0-y1) + x1**2 * (y2-y0) + x0**2 * (y1-y2)) / denom
        C = (x1 * x2 * (x1-x2) * y0 + x2 * x0 * (x2-x0) * y1 + x0 * x1 * (x0-x1) * y2) / denom

        # Find vertex
        vertex_x = -B / (2 * A)
        #print("##############")
        #print(vertex_x)
        #print("##############")

        # print(A)
        # print("************")
        # print(B)
        # print("************")
        # print(C)
        # print("************")

        # Ensure vertex in the search range
        if Left_index < vertex_x < Right_index:
            vertex_offset, vertex_flag =handel(vertex_x)
            handel_calls_in_iteration += 1  # Increment the counter
            vertex_calls += 1  # Increment vertex_calls counter
            if vertex_flag == 1:
                # print(f"Iteration {count + 1}: handel was called {handel_calls_in_iteration} times, "
                #       f"with {mid_calls} call(s) to handel(mid_index) and "
                #       f"{vertex_calls} call(s) to handel(vertex_index).")
                return vertex_x, count + 1

        # Search range adjustment based on the results of interpolation
        if vertex_x < x1:
            Right_index = x1
            Right_offset = y1
        else:
            Left_index = x1
            Left_offset = y1
        
        count += 1
 
        # Save data to Excel
        # df = pd.DataFrame(log_data)
        # df.to_excel('BLI_results_slackbus3.xlsx', index=False)

        # print(f"Iteration {count}: handel was called {handel_calls_in_iteration} times, "
        #       f"with {mid_calls} call(s) to handel(mid_index) and "
        #       f"{vertex_calls} call(s) to handel(vertex_index).")

    # if reach to the maximum
    if count >= max_iterations:
        print(f"Search terminated after {max_iterations} iterations without finding the target.")
        return None, count
    return None, count
find_Peak()

time_end = time.time()
time_sum = time_end - time_start

print(f"'handel' function was called {handel_calls} times.")

print(f"Time taken: {time_sum:.6f} seconds")
# print(f"Results have been saved to 'BLI_results_slackbus3.xlsx'")
