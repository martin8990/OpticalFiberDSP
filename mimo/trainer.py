
# Author M.C. van Leeuwen (m.c.v.leeuwen@student.tue.nl)
import numpy as np
import EvaluationFunctions.MimoEvaluation as eval

# The blockdistibuter exists to provide signal blocks to the components 
# of the mimo. This makes the components cleaner as they dont have to define the ranges on their own .
# It also contains the block settings, it therefore avoids
# dataclumps in the code. 

class Trainer():
    def __init__(self,sequence,lb,lbp,ntraining_syms):
        shift = int(-lb/2) + lbp
        self.sequence_synced = np.roll(sequence,shift,axis=1)
        self.in_training = True
        self.stop_phaserec = True
        self.lbp = lbp
        self.ntrainingblocks = ntraining_syms/lb 
        self.ntraining_syms = ntraining_syms
        self.discover_constellation_and_find_symids()
      
        
    def update_block(self,i_block,range_block):
        self.block_sequence = self.sequence_synced[:,range_block]
        if i_block>0:
            self.block_sequence_buffered = self.sequence_synced[:,range_block[0]-self.lbp : range_block[-1]+self.lbp+1]
        if i_block > self.ntrainingblocks-20:
            self.stop_phaserec = False
        if i_block>self.ntrainingblocks:
            self.in_training = False

    
    def sort_sig_per_sym(self,sig):
        sig_sym = []
        ctl = self.constellation
        symids= self.symids
        for i_mode in range(sig.shape[0]):
            sig_sym_mode = []
            for i_bit in range(len(ctl)):
                sig_sym_mode.append([])    
            for k in range(sig.shape[1]):
                my_answer = symids[i_mode,k]
                sig_sym_mode[my_answer].append(sig[i_mode,k])
            sig_sym_mode_arr = []
            for i_bit in range(len(ctl)):
                sig_sym_mode_arr.append(np.asarray(sig_sym_mode[i_bit]))

            sig_sym.append(sig_sym_mode_arr)
        return sig_sym

    def discover_constellation_and_find_symids(self):
        constellation = []
        seq = self.sequence_synced
        symids = np.zeros(seq.shape,dtype = int)
        for i_mode in range(seq.shape[0]):
            for k in range(seq.shape[1]):
                sym = seq[i_mode,k]
                if sym not in constellation:
                    constellation.append(sym)
                symids[i_mode,k] = constellation.index(sym)

        self.constellation = constellation
        self.symids = symids

    def calculate_ser_ber(self,sig):
        start = self.ntraining_syms
        stop = sig.shape[1] - 10000
        id_convert,bitmap = eval.get_8qam_map(self.constellation)
        if stop-start < 1000:
            print("Too short for SER")
        else:
            ser,ber = eval.calculate_ser_ber(sig[:,start:stop],self.symids[:,start:stop],bitmap,self.constellation,id_convert)
            print("Calculation Between " ,start," and ", stop)
            print("ser = ",eval.format_list_of_numbers(ser))
            seravg = "{:.2E}".format( np.average(np.asarray(ser)))
            print("ser_avg = ",seravg)
            ber_avg = "{:.2E}".format( np.average(np.asarray(ber)))
            print("ber = ",eval.format_list_of_numbers(ber))
            print("ber_avg = ",ber_avg)            

    
