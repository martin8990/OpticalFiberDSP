import numpy as np
import matplotlib.mlab as mlab

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

def calculate_radius_directed_error(sig : np.array,R2):
    return  R2 - abs(sig)**2 # Faruk, 2017

def calculate_impulse_response(y : np.array,x : np.array):
    Y = np.fft.fft(y)
    X = np.fft.fft(x[0:len(y)])
    return np.fft.ifft(Y/X)
    def AddTrainingLoops(self,sig : np.ndarray,ovsmpl,nloops : int):
        n_training_syms = self.n_training_syms
        training_samps = sig[:,:ovsmpl * n_training_syms]
        sig_with_loops = sig.copy()
        
        for i_loop in range(nloops):
            sig_with_loops = np.append(training_samps,sig_with_loops,axis = 1)
            self.trainingSyms = np.append(self.trainingSyms,self.trainingSyms,axis = 1)
        self.n_training_syms = n_training_syms + (nloops * n_training_syms)
        return sig_with_loops, self.trainingSyms

#### NEED QAMPY SIGNAL! ####
def calculate_BER(sig,range : np.array, synced=False,signal_rx = None):
    
        sig = sig[:,range]
        
        
        nmodes = sig.shape[0]
        syms_demod = sig.make_decision(sig[:,:])
        
        symbols_tx, syms_demod = sig._sync_and_adjust(sig.symbols, syms_demod, synced)
        # TODO: need to rename decode to demodulate
        bits_demod = sig.demodulate(syms_demod)
        tx_synced = sig.demodulate(symbols_tx)
        

        errs = np.count_nonzero(tx_synced ^ bits_demod, axis=-1)
        print(errs)
        return np.asarray(errs) / bits_demod.shape[1]


def make_decision(sig,constellation):
    dists = (np.abs(sig[:,:,np.newaxis] - constellation))**2    
    decisions = dists.argmin(axis = 2)
    return decisions
    
            

def calculate_BER_Martin(sig,answers,constellation,start, length,lb,shift=0):    
    sig = sig[:,start -int(lb/2) + shift : start+ length -int(lb/2) + shift]
    answers = answers[:,start:start +length]
    print(sig.shape)
    decisions = make_decision(sig[:,:],constellation)
    errs = np.count_nonzero(answers - decisions, axis=-1)
    print(errs)
    return np.asarray(errs) / answers.shape[1]

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