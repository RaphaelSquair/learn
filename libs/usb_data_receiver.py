#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import serial
from dataclasses import dataclass

ser = serial.Serial(port = "COM7",baudrate = 115200 ,timeout=1,bytesize=8, stopbits=serial.STOPBITS_ONE)

data = ""                           # Used to hold data coming over UART
icarga_a=0
icarga_b=0
iconv_a=0
iconv_b=0
vca=0
vab=0
vbc=0
vcplint=0
irefint=0
tx1vis=0

Divider = 32768.0;
gainv = 3.0;
gaini = 2.0;
offset = 0.1;
vref = 1.2;
iref = 1.667;
vbase = 1.200;
ibase = 1.667;

Freq = 1000000;

vpu1=0;
internal=0;
internal1=0;
out=0;
out1=0;
outRload = 0;
Iin = 0;
Iout = 0;
er1 = 0;

Vdc_Base = 450.0 	
Vdc_Ref = 450.0			
V_Base = 311.0
I_Base = 16.36

ganho_iout = 32768
ganho_vbc_Q12 =4096*(1300.0/V_Base)
ganho_vdc_Q12 =4096*(1200.0/Vdc_Base)
ganho_vab_Q12 =4096*(1300.0/V_Base)
ganho_i_Q12 =4096*(100.0/I_Base)

@dataclass
class INPUT_IO:
    IA: int = 1
    IB: int = 1
    IC: int = 1
    ID: int = 1
    IE: int = 1

@dataclass
class OUTPUT_IO:
    OG: int = 1
    OH: int = 1
    OI: int = 1
    OJ: int = 1
    OK: int = 1
    OL: int = 1


input_IO = INPUT_IO
#output_IO = np.bit



T = 1/Freq;
b1 = T-0.4;
b0 = T+0.4;
a1 = T-0.08;
a0 = T+0.08;

z1 = -10;
z0 = 10;
c1 = T-10;
c0 = T+10;

Rx1 = [0,0,0,0,0,0,0]

while(1):

    # Wait until there is data waiting in the serial buffer
    if(ser.in_waiting > 0):

        # Read data out of the buffer until a carraige return / new line is found
        c = ser.read(15)        # read up to ten bytes (timeout)
        # Print the contents of the serial data
        print (c)

        Tx1_1=c[0];
        Tx1_2=c[1];
        Tx2_1=c[2];
        Tx2_2=c[3];
        Tx3_1=c[4];
        Tx3_2=c[5];
        Tx4_1=c[6];
        Tx4_2=c[7];
        Tx5_1=c[8];
        Tx5_2=c[9];
        Tx6_1=c[10];
        Tx6_2=c[11];
        Tx7_1=c[12];
        Tx7_2=c[13];

        Tx1_AD = ((((Tx1_1))&0xFF)+(((Tx1_2)<<8)&0xFF00))
        Tx2_AD = ((((Tx2_1))&0xFF)+(((Tx2_2)<<8)&0xFF00))        
        Tx3_AD = ((((Tx3_1))&0xFF)+(((Tx3_2)<<8)&0xFF00))
        Tx4_AD = ((((Tx4_1))&0xFF)+(((Tx4_2)<<8)&0xFF00))
        Tx5_AD = ((((Tx5_1))&0xFF)+(((Tx5_2)<<8)&0xFF00))
        Tx6_AD = ((((Tx6_1))&0xFF)+(((Tx6_2)<<8)&0xFF00))
        Tx7_AD = ((((Tx7_1))&0xFF)+(((Tx7_2)<<8)&0xFF00))
        
        input_IO.IA=((c[14])&0X1);
        input_IO.IB=((c[14]>>1)&0X1);
        input_IO.IC=((c[14]>>2)&0X1);
        input_IO.ID=((c[14]>>3)&0X1);
        input_IO.IE=((c[14]>>4)&0X1);

        print(Tx1_1)
        print(Tx1_AD)

        vcplint=Tx1_AD ;
        irefint=Tx2_AD ;

        vcpl = ((vcplint/Divider)-offset)*gainv;
        icpl = ((irefint/Divider)-offset)*gaini;

        vpu = vcpl/vbase;
        ipu = icpl/ibase;

        er= (1-vpu);
        Rload =(vpu/abs(ipu));
        if(Rload>=10.25):
            Rload = 10.25
        elif(Rload<=0):
            Rload = 0.01
        

        internal = ((er*a0 +er1*a1 -internal1*b1)/b0);
        out = (internal*c0 +internal1*c1 -out1*z1)/z0;
        outRload = er*Rload*0.002;

        Iin = out-outRload;

        
        vpu1 = vpu;
        internal1 = internal;
        out1 = out;
        er1 =er;

        vab_pu= (int((Iin*ganho_iout))+1500);                   
        vbc_pu= (int(int(vbc * ganho_vbc_Q12)>>8));                   
        vca_pu= (int(int(vca * ganho_vab_Q12)>>8));                  

        vab_send =(((vab_pu)));
        vbc_send =(((vbc_pu)));
        vca_send =(((vca_pu)));

        pwm_on = 1;
        if (pwm_on==1):       
            pv1 = (int)(vab_send&0xFF);
            pv2 = (int)(((vab_send)>>8)&0xFF);
            pv3 = (int)(vbc_send);
            pv4 = (int)((vbc_send)>>8);
            pv5 = (int)(vca_send);
            pv6 = (int)((vca_send)>>8);
            #pv7 = (int)(output_IO);
        else:
            pv1 = 0;
            pv2 = 0;
            pv3 = 0;
            pv4 = 0;
            pv5 = 0;
            pv6 = 0;
        
        Rx1[0] = (pv1);
        Rx1[1] = (pv2);
        Rx1[2] = (pv3);
        Rx1[3] = (pv4);
        Rx1[4] = (pv5);
        Rx1[5] = (pv6);
        Rx1[6] = 16;#(output_IO.view(np.uint8));

        ser.write(Rx1)
