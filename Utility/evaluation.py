import numpy as np
import matplotlib.mlab as mlab
import cmath as cm
def calculate_final_error(err : list,limit_lower,limit_upper)-> list:
    list_final_error = []
    
    for i_mode in range(len(err)):
        list_final_error.append(np.average(err[i_mode][limit_lower:limit_upper]))
    return list_final_error

def calculate_convergence(err : list,final_error) -> list:
    
    list_convergence = []
    for i_mode in range(len(err)):
        trigger = final_error[i_mode]
        for k in range(0,len(err[i_mode])-1):
            if err[i_mode][k] < trigger:
                #print("k ",k,"trig ",trigger , "val ",err[i_mode][k])
                break
        list_convergence.append(k)
    return list_convergence

    ber = np.asarray(bit_errs) / sig_bits.shape[1]
    return ser,ber

def get_8qam_map(constellation):
    anglo_const = [np.arctan2(i.real,i.imag) for i in constellation]
    id_convert = np.argsort(np.asarray(anglo_const))
    bitmap = [(1,1,1),(0,1,1),(1,1,0),(0,1,0), (1,0,0),(0,0,0),(1,0,1),(0,0,1)]
    return id_convert,bitmap




def seperate_per_bit(constellation,answers,sig,lb,shift = 0):
    bit_sigs = []
    for i_mode in range(sig.shape[0]):
        mode_bit_sigs = []
        for i_bit in range(len(constellation)):
            mode_bit_sigs.append([])    
        for k in range(int(lb/2),sig.shape[1]):
            my_answer = answers[i_mode,k]
            mode_bit_sigs[my_answer].append(sig[i_mode,k-int(lb/2) + shift])
        mode_bit_sigs_arr = []
        for i_bit in range(len(constellation)):
            mode_bit_sigs_arr.append(np.asarray(mode_bit_sigs[i_bit]))

        bit_sigs.append(mode_bit_sigs_arr)
    return bit_sigs

def derive_constellation_and_answers(sequence):
    constellation = []
    answers = np.zeros(sequence.shape,dtype = int)
    for i_mode in range(sequence.shape[0]):
        for k in range(sequence.shape[1]):
            sym = sequence[i_mode,k]
            if sym not in constellation:
                constellation.append(sym)
            answers[i_mode,k] = constellation.index(sym)
    return constellation,answers

def AddTrainingLoops(sig,sequence,ovsmpl,nloops,syms_per_loop):
    for i_loop in range(nloops):
        sig = np.append(sig[:,:syms_per_loop*ovsmpl],sig,axis = 1)
        sequence = np.append(sequence[:,:syms_per_loop],sequence,axis = 1)

    return sequence,sig

def make_decision(sig,constellation):
    dists = (np.abs(sig[:,:,np.newaxis] - constellation))**2    
    decisions = dists.argmin(axis = 2)
    return decisions
    
            

def calculate_ser(sig,answers,constellation):    
    decisions = make_decision(sig[:,:],constellation)
    errs = np.count_nonzero(answers - decisions, axis=-1)
    print(errs)
    return np.asarray(errs) / answers.shape[1]

def decode(decisions,map,id_convert):

    bps = len(map[0])      # Bits per symbol 
    bits = np.zeros((decisions.shape[0],decisions.shape[1],bps))
    for i_mode in range(decisions.shape[0]):
        for k in range(decisions.shape[1]):

            mybits = map[id_convert[decisions[i_mode,k]]]
            for bit in range(len(mybits)):
                bits[i_mode,k,bit] = mybits[bit]
    return bits.reshape(decisions.shape[0],decisions.shape[1]*bps)
            
def format_list_of_numbers(list):
    strs = ['{:.2E}'.format(num) for num in list]
    return strs

def calculate_ser_ber(sig,answers,decodation_map,constellation,id_convert):    
    decisions = make_decision(sig[:,:],constellation)
    sym_errs = np.count_nonzero(answers - decisions, axis=-1)
    print("symbol errors : ",sym_errs)
    ser = np.asarray(sym_errs) / answers.shape[1]
    
    sig_bits = decode(decisions,decodation_map,id_convert)
    seq_bits = decode(answers,decodation_map,id_convert)
    
    bit_errs = np.count_nonzero(sig_bits - seq_bits, axis=-1)
    print("bit errors : ",bit_errs)
    ber = np.asarray(bit_errs) / sig_bits.shape[1]
    return ser,ber