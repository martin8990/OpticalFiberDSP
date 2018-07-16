import numpy as np


import pandas as pd
from Testing.PandaTest.SignalCreator import *
from Testing.PandaTest.DirectoryCreator import *



print("initialized")
testDir = create_dir_for_dsp_evaluation()
movavg_taps = 1000
df = pd.DataFrame()
df = EvaluateSignals(df,testDir)
       
#df = df.sort_values(['mode','pmd','snr','ber'],ascending=[0,0,1,1])
print(df)
df.to_csv(testDir + "\Results.csv", sep=',')

