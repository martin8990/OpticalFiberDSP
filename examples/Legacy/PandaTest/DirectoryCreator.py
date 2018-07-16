import os
from datetime import datetime

def create_dir_for_dsp_evaluation():

    time_string = str(datetime.now())
    time_string = time_string.replace(":","-")
    time_string = time_string.replace(" ","_")
    directory = os.getcwd() + "\TestResults\Test_" +  time_string
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def create_dir_for_signal(basedir,PMD,SNR,mixing,impulse_impaired : bool,ovsmpl : int,rep : int,label = ""):
    title = label + "PMD" + str(PMD) + "_SNR" + str(SNR) + "_MIX" + str(mixing)  + "_impulse_impaired" + str(impulse_impaired) + "_ovsmpl" + str(ovsmpl) + "_rep" + str(rep)
    print(title)
    directory = basedir + "\\" + title
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def create_dir_for_dict(basedir,dict):
    keys = list(dict.keys())
    title = ""
    for i_key in range(len(keys)):
        title = title + "_" + keys[i_key] + str(dict[keys[i_key]])
    print(title)
    directory = basedir + "\\" + title
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def create_dir_for_signalrow(basedir,dataframe,i_row):
    collums = list(dataframe.columns.values)
    title = ""
    for i_col in range(len(collums)):
        title = title + "_" + collums[i_col] + str(dataframe[collums[i_col]][i_row])
    print(title)
    directory = basedir + "\\" + title
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory



def create_dir_for_mimo_result(basedir,mu,ntaps,label : str):
    title = label + "_mu" + str(mu) + "_ntaps" + str(ntaps)
    print(title)
    directory = basedir + "\\" + title
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def CreateLogFile(BER,dir):
    f= open(dir + "\\Log.txt","w+")
    f.write("BER : " + str(BER))


if __name__ == "__main__":
    #basedir = create_dir_for_dsp_evaluation()
    #sampledir = create_dir_for_testcase(1e-12,1e-12,1e-12,1e-1,1e-8,1e-6,basedir,"NoImpulse")
    #print(sampledir)
    print("HI")