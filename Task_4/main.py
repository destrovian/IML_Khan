import pandas as pd
import numpy as np
import Matching

def matching(intA, intB, intC):
    tempB =0
    tempC =0
    for i in range(0,20):
        for j in range(0,20):
            if df.loc[intA,i*2]==df.loc[intB,j*2]:
                tempB = tempB + df.loc[intA,i*2+1] + df.loc[intB,j*2+1]
            if df.loc[intA,i*2]==df.loc[intC,j*2]:
                tempC = tempC + df.loc[intA,i*2+1] + df.loc[intB,j*2+1]
    return tempB>tempC

df = pd.read_csv("classification.csv",header=None)
print(df)

print(matching(0,6,9))
