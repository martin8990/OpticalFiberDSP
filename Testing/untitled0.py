# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:06:15 2018

@author: marti
"""

def cal_L_per_km(E,km):
    L = E/((1000*h) * 1/(rho_fuel))
    return 100/(km/L)


E_NEDC = 37.8e3  
E_NEDC_quad = E_NEDC - 2150
E_NEDC_heur = E_NEDC_Total - 1750

E_FTP = 55.5e3
E_FTP_quad = E_FTP - 2050
E_FTP_heur = E_FTP - 1745

E_HWFET = 23.3e3
E_HWFET_quad = H_HWFET - 2990
E_HWFET_heur = H_HWFET - 2799
 

km = 0.95871
h = 42500
rho_fuel = 0.72

l100_FTP = cal_L_per_km(E_FTP,km)
l100_QUAD_FTP = cal_L_per_km(E_FTP_quad,km)
l100_Heur_FTP = cal_L_per_km(E_FTP_heur,km)

km = 1.37293
l100_NEDC = cal_L_per_km(E_NEDC,km)
l100_QUAD_NEDC = cal_L_per_km(E_NEDC_quad,km)
l100_Heur_NEDC = cal_L_per_km(E_NEDC_heur,km)
#km = 
#l100_HWFET = cal_L_per_km(E_HWFET,km)
#l100_QUAD_HWFET = cal_L_per_km(E_HWFET_quad,km)
#l100_Heur_HWFET = cal_L_per_km(E_HWFET_heur,km)



