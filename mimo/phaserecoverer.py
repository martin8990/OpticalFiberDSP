import numpy as np
from mimo.mimo import BlockDistributer,Trainer

# TODO : Class is to long, should be refactored
class BlindPhaseSearcher():
    def __init__(self,block_distr : BlockDistributer,trainer : Trainer,num_testangles,search_area = np.pi/2):
    
        b = np.arange(num_testangles)
        self.angles = b/num_testangles * search_area
        self.i_block = 0
        self.num_testangles =num_testangles
        self.lbp = trainer.lbp
        self.sa = search_area
        nmodes = block_distr.nmodes
        nblocks = block_distr.nblocks
        self.buffer = np.ones((nmodes,trainer.lbp*2),dtype = np.complex128) * trainer.constellation[0]
        
        self.phase_collection = []
        self.slips_up = []
        self.slips_down = []

        for i_mode in range(nmodes):
            self.phase_collection.append([0])
            self.slips_up.append([0])
            self.slips_down.append([0])
        
        self.prev_phases = np.ones(nmodes) * 0
        self.counter = np.zeros(nmodes)
        self.i_block = 0
                   
    def __denoise(self, nearest_dist_per_angle):
        csum = np.cumsum(nearest_dist_per_angle, axis=0)
        nearest_dist_per_angle_denoised = csum[2*self.lbp:]-csum[:-2*self.lbp] 
     
        return nearest_dist_per_angle_denoised
        
    def __get_angle_id_with_nearest_distance(self,nearest_dist_per_angle_denoised):
        return nearest_dist_per_angle_denoised.argmin(1)

    def __find_best_decisions(self,i_mode,block_distr : BlockDistributer,block,trainer : Trainer):
        angles = self.angles
        constellation = trainer.constellation
                    
        sig_rotated = block[:,np.newaxis]*np.exp(1j*angles)
        distances = abs(sig_rotated[:, :, np.newaxis]-constellation)**2

        nearest_dist_per_angle = distances.min(axis=2)
        nearest_dist_per_angle_denoised = self.__denoise(nearest_dist_per_angle)
        
        decisions = self.__get_angle_id_with_nearest_distance(nearest_dist_per_angle_denoised)
        return decisions

    def __select_angles(self,angles, decisions):
        nsamps = decisions.shape[0]
        chosen_angles = np.zeros(nsamps)
        for i in range(nsamps):
            chosen_angles[i] = angles[decisions[i]]
        return chosen_angles

    def remove_cycle_slips(self, i_mode, lb, phases_mode):
        inert_phases = np.zeros_like(phases_mode)
        sa = self.sa
        dsa = self.sa/2
        for k,cur_phase in enumerate(phases_mode):
             cur_phase += sa*self.counter[i_mode]
             delta_phase = cur_phase - self.prev_phases[i_mode] 
             if delta_phase > dsa:
                self.slips_up[i_mode].append(k + lb * self.i_block)
                self.counter[i_mode] -= 1
                cur_phase -= sa
             elif delta_phase < -dsa:
                self.slips_down[i_mode].append(k + lb * self.i_block)
                self.counter[i_mode] += 1
                cur_phase += sa
             self.prev_phases[i_mode] = cur_phase
             inert_phases[k] = cur_phase
        return inert_phases

    def peek_phase_offset(self,block,i_mode,trainer : Trainer):
        
        seq = trainer.block_sequence[i_mode]
        angles_sequence = np.arctan2(seq.imag,seq.real)
        angles_block = np.arctan2(block.imag,block.real)
        phases = angles_sequence-angles_block
        
        return phases
      

    def recover_phase(self,block_distr : BlockDistributer,trainer: Trainer):
        phases = []
        block = block_distr.block_compensated
        lb = block_distr.lb
        lbp = trainer.lbp

        for i_mode in range(block.shape[0]):
            block_appended = np.append(self.buffer[i_mode],block[i_mode])
            self.buffer[i_mode] = block[i_mode,-self.lbp*2:]
            block[i_mode] = block_appended[self.lbp:-self.lbp]
            if trainer.in_training:
                if self.i_block == 0:
                    phases_mode = np.zeros(lb)
                else :
                    phases_mode = self.peek_phase_offset(block[i_mode],i_mode,trainer)
                    phases_mode = np.average(phases_mode) * np.ones_like(phases_mode)
            else:
                decisions =  self.__find_best_decisions(i_mode,block_distr,block_appended,trainer)
                phases_mode = self.__select_angles(self.angles,decisions)
            phases_mode = self.remove_cycle_slips(i_mode, lb, phases_mode)
            phases.append(phases_mode)
            self.phase_collection[i_mode].extend(phases_mode)
                   
        phases = np.asarray(phases)
        self.i_block+=1
        block_phaserec = block*np.exp(1j*phases)
        
        block_distr.insert_compensated_block(block_phaserec)
        block_distr.shift_fd_block(-trainer.lbp)
        block_distr.phases = phases
            


        




    
