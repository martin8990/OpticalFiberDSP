import numpy as np
from mimo.mimo import BlockDistributer

def compensate(block_distr : BlockDistributer,H):
    nmodes = block_distr.nmodes
    ovsmpl = block_distr.ovsmpl
    lb = block_distr.lb
    second_block = range(lb,2 * lb)
    
    double_block_fd = block_distr.double_block_fd
    block_compensated = np.zeros(nmodes * lb ,dtype = np.complex128).reshape(nmodes,lb)

    for i_input in range(nmodes):
        for i_output in range(nmodes):        
            FD_temp = np.zeros(lb*2,dtype = np.complex128)  
            for i_ovsmpl in range(ovsmpl):
                FD_temp += double_block_fd[i_input,i_ovsmpl] * H[i_input,i_output,i_ovsmpl]
            block_compensated[i_output]  += np.fft.ifft(FD_temp)[second_block]
    block_distr.insert_compensated_block(block_compensated)

def compensate_widely(block_distr : BlockDistributer,H):
    nmodes = block_distr.nmodes
    ovsmpl = block_distr.ovsmpl
    lb = block_distr.lb
    second_block = range(lb,2 * lb)
    
    double_block_fd = block_distr.double_block_fd
    block_compensated = np.zeros(nmodes * lb ,dtype = np.complex128).reshape(nmodes,lb)

    for i_input in range(nmodes):
        for i_output in range(nmodes):        
            FD_temp = np.zeros(lb*2,dtype = np.complex128)  
            for i_ovsmpl in range(ovsmpl):
                for i_wide in [0,1]:
                    FD_temp += double_block_fd[i_input,i_ovsmpl,i_wide] * H[i_input,i_output,i_ovsmpl,i_wide]
            block_compensated[i_output]  += np.fft.ifft(FD_temp)[second_block]
    block_distr.insert_compensated_block(block_compensated)