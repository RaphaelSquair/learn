import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wisardpkg as wp
import scipy.fftpack
import copy
import sympy.combinatorics.graycode as gray


def rolling_window(x, window):
    shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

#def term(x,res,):
def fftsig(data,h):
    fftdata = copy.copy(data)
    L = len(fftdata)
    Fs = 1 / h

    f_fft = ((Fs / 1) * np.arange(0, (L / 2), 1)) / L  # vetor de frequencias a ser passado para fft

    Y = scipy.fftpack.fft(fftdata / max(abs(fftdata)))  # não entendi
    P2 = np.abs(Y / L)
    fft = P2[0:int(L / 2)]
    fft[1:-2] = 2 * fft[1:-2]
    f_fft = f_fft[:len(fft)]
    return [f_fft,fft]

def dec2gray(x,res):
    return gray.bin_to_gray(bin(int((2**res)*x)).replace("0b",""))

def gray2dec(x,res):
    return float(int(gray.gray_to_bin(str(int(x))),2))/(2**res)
def mse(x,y):
    return sum((x-y)**2)
'''
algoritimo de regressão ok

busca de hiperparametros:
    res             5-12
    window_size     5-30
    addressSize     5-20
    

'''


#data =pd.read_csv("lowres.csv",sep=",")
#data =pd.read_csv("newdata.csv",sep=";")

tensoes =pd.read_csv("1khz/tensoes.csv",sep=",")
correntes = pd.read_csv("1khz/correntes.csv",sep=",")

res =14             #best 10
window_size = 6    #best 5
addressSize = 13    #best 10

Ioffsetout  =+0.5
Ibaseout    =1
Vbase       =1
Ibase       =1
Toffset     =4.999         # os dados começam em 5s
Voffset     =-0.5      # A Tensão tem média em 1 depois de normalizada
Ioffset     =+0.5      # A corrente tem amplitude maxima de 0.4 e oscila a baixo de 0

meanvalidation =10

def func(x,tensoes,correntes):
    res = int(x[0])
    window_size = int(x[1])
    addressSize = int(x[2])
    time        = (copy.copy(tensoes.values[:,0])-Toffset)[:-3]
    #gvsc_V     = ((copy.copy(tensoes.values[:,1])/Vbase)+Voffset)[:-3]
    cpl_V       = ((copy.copy(tensoes.values[:,2])/Vbase)+Voffset)[:-3]
    dcstatcom_I = ((copy.copy(correntes.values[:,1])/Ibase)+Ioffset)[:-3]
    #ref_I      = ((copy.copy(correntes.values[:,2])/Ibase)+Ioffset)[:-3]
    #gvsc_I     = ((copy.copy(correntes.values[:,3])/Ibase)+Ioffset)[:-3]
    grid_I      = ((copy.copy(correntes.values[:,4])/Ibase)+Ioffset)[:-3]

    time_N = time[:int(len(time)/2)]
    cpl_V_N = cpl_V[:int(len(time)/2)]
    dcstatcom_I_N = dcstatcom_I[:int(len(time)/2)]
    grid_I_N = grid_I[:int(len(time)/2)]

    time_Y = time[int(len(time)/2):]-15
    cpl_V_Y = cpl_V[int(len(time)/2):]
    dcstatcom_I_Y = dcstatcom_I[int(len(time)/2):]
    grid_I_Y = grid_I[int(len(time)/2):]

    gray_time       = copy.copy(time_Y)
    #gray_gvsc_V    = gvsc_V
    gray_cpl_V      = copy.copy(cpl_V_Y)
    gray_dcstatcom_I= copy.copy(dcstatcom_I_Y)
    #gray_ref_I     = ref_I
    #gray_gvsc_I    = gvsc_I
    gray_grid_I     = copy.copy(grid_I_Y)




    for i in range(len(gray_time)):
        #gray_time[i] = dec2gray(gray_time[i],res)
        #gray_gvsc_V[i] = dec2gray(gray_gvsc_V[i],res)
        gray_cpl_V[i] = dec2gray(gray_cpl_V[i],res)
        #gray_gvsc_I[i] = dec2gray(gray_gvsc_I[i],res)
        gray_grid_I[i] = dec2gray(gray_grid_I[i],res)
        gray_dcstatcom_I[i] = dec2gray(gray_dcstatcom_I[i], res)
        #gray_ref_I[i] = dec2gray(gray_ref_I[i], res)


    #gray_time_tapped        = rolling_window(gray_time,window_size)
    #gray_gvsc_V_tapped      = rolling_window(gray_gvsc_V,window_size)
    gray_cpl_V_tapped       = rolling_window(gray_cpl_V,window_size)
    #gray_gvsc_I_tapped      = rolling_window(gray_gvsc_I,window_size)
    gray_grid_I_tapped      = rolling_window(gray_grid_I,window_size)
    #gray_dcstatcom_I_tapped = rolling_window(gray_dcstatcom_I,window_size)
    #gray_ref_I_tapped       = rolling_window(gray_ref_I,window_size)

    #gray_gvsc_V_tapped_wp = np.empty(shape=[len(gray_time_tapped)],dtype=object)
    gray_cpl_V_tapped_wp = np.empty(shape=[len(gray_cpl_V_tapped)],dtype=object)
    #gray_gvsc_I_tapped_wp = np.empty(shape=[len(gray_time_tapped)],dtype=object)
    gray_grid_I_tapped_wp = np.empty(shape=[len(gray_grid_I_tapped)],dtype=object)
    #gray_dcstatcom_I_tapped_wp = np.empty(shape=[len(gray_dcstatcom_I_tapped)],dtype=object)

    input_wp = np.empty(shape=[len(gray_grid_I_tapped)],dtype=object)


    for i in range(len(gray_grid_I_tapped)):
        for j in range(len(gray_grid_I_tapped[0])):
            if j==0:
                gray_cpl_V_tapped_wp[i] = np.array([int(d) for d in str(int(gray_cpl_V_tapped[i][j])).zfill(res)])
                gray_grid_I_tapped_wp[i] = np.array([int(d) for d in str(int(gray_grid_I_tapped[i][j])).zfill(res)])
               # gray_dcstatcom_I_tapped_wp[i] = np.array([int(d) for d in str(int(gray_dcstatcom_I_tapped[i][j])).zfill(res)])
                input_wp[i] = np.append(gray_cpl_V_tapped_wp[i],gray_grid_I_tapped_wp[i])
            else:
                gray_cpl_V_tapped_wp[i] = np.append(gray_cpl_V_tapped_wp[i],[int(d) for d in str(int(gray_cpl_V_tapped[i][j])).zfill(res)])
                gray_grid_I_tapped_wp[i] = np.append(gray_grid_I_tapped_wp[i],[int(d) for d in str(int(gray_grid_I_tapped[i][j])).zfill(res)])
                #gray_dcstatcom_I_tapped_wp[i] = np.append(gray_dcstatcom_I_tapped_wp[i],[int(d) for d in str(int(gray_dcstatcom_I_tapped[i][j])).zfill(res)])
                input_wp[i] = np.append(input_wp[i],gray_cpl_V_tapped_wp[i])
                input_wp[i] = np.append(input_wp[i], gray_grid_I_tapped_wp[i])
        input_wp[i]=list(input_wp[i])
        if ((i % (round(len(gray_grid_I_tapped) / 1000))) == 0):
            print(round(i / (len(gray_grid_I_tapped) / 100), 2))


    input_wp1 = list(input_wp[:-1])
    output_wp1 =list(dcstatcom_I_Y[window_size:])

    output_wp_train = output_wp1[:-(int(len(output_wp1)/3)+1)]
    output_wp_test = output_wp1[int(len(output_wp1)*2/3):]
    input_wp_train = input_wp1[:len(output_wp_train)]
    input_wp_test = input_wp1[len(output_wp_train):]


    ds = wp.DataSet(input_wp_train,output_wp_train)
    outhist = np.zeros((meanvalidation,len(input_wp_test)))
    errorhist = np.arange(meanvalidation,dtype=float)
    for k in range(meanvalidation):
        mean = wp.PowerMean(1)

        rwsd = wp.RegressionWisard(
           addressSize,                # required
           orderedMapping=False,       # optional
           completeAddressing=True,    # optional
           mean=mean,                  # optional
        )

        rwsd.train(ds)
        xpred = wp.DataSet(input_wp_test)
        out = rwsd.predict(xpred)
        if(len(out)>len(output_wp_test)):
            out = out[:len(output_wp_test)]
        elif(len(out)<len(output_wp_test)):
            output_wp_test = output_wp_test[:len(out)]
        desnormalizedout = (np.array(out)-Ioffsetout)*Ibaseout
        desnormalizedout_wp_test = (np.array(output_wp_test)-Ioffset)*Ibase
        error = mse(desnormalizedout, desnormalizedout_wp_test)
        #outhist[k] = desnormalizedout
        errorhist[k] = error


    return [desnormalizedout,desnormalizedout_wp_test,error,outhist,errorhist]

[ind01out,controlout,error,outhist,errorhist] = func([res,window_size,addressSize],tensoes,correntes)

errormean = np.mean(errorhist)
errorvar  = np.var(errorhist)





fig1=plt.figure(1)
plt.plot(np.arange(len(controlout)),ind01out)
plt.plot(np.arange(len(controlout)),controlout)


plt.show()

#mse(controlout,ind01out)


print("end")
