import numpy as np
from mimo.mimo import BlockDistributer

def compensate(block_distr : BlockDistributer,H):
    """
    Applies the learned taps to the current block of samples, works for oversampled and widely linear signals 
    Parameters
    ----------
    H : np.ndarray, Frequency domain taps 
    """
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
                for i_ovconj in range(block_distr.ovconj):
                    FD_temp += double_block_fd[i_input,i_ovsmpl,i_ovconj] * H[i_input,i_output,i_ovsmpl,i_ovconj]
            block_compensated[i_output] += np.fft.ifft(FD_temp)[second_block]
    block_distr.insert_compensated_block(block_compensated)