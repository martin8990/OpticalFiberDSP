import numpy as np
# Timo Pfau et al,
#  Hardware-Efficient Coherent Digital Receiver Concept With Feedforward Carrier Recovery for M-QAM Constellations
#  , Journal of Lightwave Technology 27, pp 989-999 (2009)

def __denoise(lb, nearest_dist_per_angle):
    csum = np.cumsum(nearest_dist_per_angle, axis=0)
    nearest_dist_per_angle_denoised = csum[2*lb:]-csum[:-2*lb] 
    return nearest_dist_per_angle_denoised

def __get_angle_id_with_nearest_distance(nearest_dist_per_angle_denoised):
    return nearest_dist_per_angle_denoised.argmin(1)

def __find_best_decisions(sig : np.array, angles, constellation, lb):
    decisions = np.zeros(sig.shape[0], dtype=np.int)
    
    sig_rotated = sig[:,np.newaxis]*np.exp(1j*angles)
    distances = abs(sig_rotated[:, :, np.newaxis]-constellation)**2

    nearest_dist_per_angle = distances.min(axis=2)
    nearest_dist_per_angle_denoised = __denoise(lb, nearest_dist_per_angle)
    print(nearest_dist_per_angle_denoised.shape)
    
    decisions[lb:-lb] = __get_angle_id_with_nearest_distance(nearest_dist_per_angle_denoised)
    return decisions

def __select_angles(angles, decisions):
    nsamps = decisions.shape[0]
    chosen_angles = np.zeros(nsamps, dtype=np.float64)
    for i in range(nsamps):
        chosen_angles[i] = angles[decisions[i]]
    return chosen_angles


def blind_phase_search(sig, num_testangles, constellation, lb):
   
    b = np.arange(num_testangles)
    angles = b/num_testangles * np.pi/2
    phases = []
    for i_mode in range(sig.shape[0]):
        decisions =  __find_best_decisions(sig[i_mode], angles, constellation, lb)
        phases.append(__select_angles(angles,decisions))        
    phases = np.asarray(phases)
    phases = np.unwrap(phases)
    return sig*np.exp(1.j*phases)




    