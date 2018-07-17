import numpy as np
## Params
import examples.capture_loader as load
import examples.equalizer as eq
import examples.settings_QAM as set

mimosettings = set.MimoSettings()
update_settings = set.UpdateSettings()
phase_recovery_settings = set.PhaseRecoverySettings()

nsyms = 5 * 10**4
sequence,sig = load.load_64Qam_best(nsyms)
sequence /=7
sig /=7

eq.equalize(sig,sequence,mimosettings,update_settings,phase_recovery_settings,showplots=True)

