
# Author M.C. van Leeuwen (m.c.v.leeuwen@student.tue.nl)
import numpy as np
import utility.evaluation as eval
import matplotlib.mlab as mlab

# Provides training symbols
class Trainer():
    def __init__(self,sequence : np.ndarray,lb,lbp,ntraining_syms,phaserec_start):
        """
        Parameters
        ----------
        Sequence : Transmitted symbols complementary to the signal
        lb : blocksize of the equalizer
        lbp : block size of the internal phase recovery
        ntraining_syms : symbols per training loop
        """
        
        shift = int(-lb/2) + lbp
        self.sequence_synced = np.roll(sequence,shift,axis=1)  
        
        self.in_training = True
        self.stop_phaserec = True
        
        self.lbp = lbp
        self.ntrainingblocks = ntraining_syms/lb 
        self.ntraining_syms = ntraining_syms
        self.i_block_pr_start = phaserec_start/lb
        self.nloops = 1        
        self.i_errorcalc = 0
      
        
    def update_block(self,i_block,range_block):
        """
        
        """
        self.block_sequence = self.sequence_synced[:,range_block]
        #if i_block>0:
        #    self.block_sequence_buffered = self.sequence_synced[:,range_block[0]-self.lbp : range_block[-1]+self.lbp+1]
        if i_block > self.i_block_pr_start:
            self.stop_phaserec = False
        if self.in_training and i_block>self.ntrainingblocks * (self.i_errorcalc+1):
            self.i_errorcalc+=1
            if self.nloops == self.i_errorcalc :
                self.in_training = False                            

    def get_error_calculator(self,block_distr):
        return self.errorcalcs[self.i_errorcalc]
    def retrieve_error(self):
        error = self.errorcalcs[0].error
        for ecalc in self.errorcalcs:
            for i_mode in range(error.shape[0]):
                error[i_mode] = np.maximum(abs(ecalc.error[i_mode]),abs(error[i_mode]))

        err_Averaged = []
        movavg_taps = 1000
        for i_mode in range(error.shape[0]):
            err_Averaged.append(mlab.movavg(error[i_mode],movavg_taps))
        return err_Averaged
        
    ## Used for making colored plots
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

          
       
    def AddTrainingLoop(self,sig,ovsmpl):
        seq = self.sequence_synced
        nsyms = self.ntraining_syms
        self.sequence_synced = np.append(seq[:,:nsyms],seq,axis = 1)
        self.nloops+=1
        print(nsyms)
        return np.append(sig[:,:nsyms*ovsmpl],sig,axis = 1)
        

    def discover_constellation_and_find_symindexes(self):
       
        seq = self.sequence_synced
        symids = np.zeros(seq.shape,dtype = int)
        for i_mode in range(seq.shape[0]): # cannot be vectorized further
            constellation,symids[i_mode] = np.unique(seq[i_mode],return_inverse=True) 
        self.constellation = constellation
        self.symids = symids

    def set_errorcalcs(self,errorcalcs : list):
        self.errorcalcs = errorcalcs
        if len(errorcalcs) != self.nloops+1 :
            raise ValueError("Need " + str(self.nloops+1)  + "but received " + str( len(errorcalcs)) + " errorcalculators" )

    def calculate_ser_ber(self,sig):
        start = self.ntraining_syms * self.nloops
        stop = sig.shape[1] - 10000
        constellation = self.constellation
        decisions = eval.make_decisions(sig,constellation)
        self.decisions = decisions
        if stop-start < 1000:
            print("Too short for SER")
        else:
            symids = self.symids[:,start:stop]
            decisions = decisions[:,start:stop]            
            ser = eval.calculate_ser(decisions,symids)
            print("Calculation Between " ,start," and ", stop)
            print("ser = ",eval.format_list_of_numbers(ser))
            seravg = "{:.2E}".format( np.average(np.asarray(ser)))
            print("ser_avg = ",seravg)
            
            if len(constellation) == 8: # 8QAM
                bitmap = eval.get_8qam_map(constellation)
                print(len(bitmap))
                ber = eval.calculate_ber(decisions,symids,bitmap)
                ber_avg = "{:.2E}".format( np.average(np.asarray(ber)))
                print("ber = ",eval.format_list_of_numbers(ber))
                print("ber_avg = ",ber_avg)
