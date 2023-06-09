#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:39:15 2023

@author: louis-yann
"""

#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import sys, math, numpy as np, matplotlib.pyplot as plt

import dynamic as dyn ,utils as ut

def plot_thrust(P, filename=None):
    figure = ut.prepare_fig(None, f'Poussée {P.name}')
    U = [0, 1., 0, 0]
    hs, machs = np.linspace(3000, 11000, 5), np.linspace(0.5, 0.8, 30)
    for h in hs:
        thrusts = [dyn.propulsion_model([0, h, dyn.va_of_mach(mach, h), 0, 0, 0], U, P) for mach in machs] 
        plt.plot(machs, thrusts)
    ut.decorate(plt.gca(), f'Poussée maximum {P.eng_name}', 'Mach', '$N$', 
                [f'{_h} m' for _h in hs])
    ut.savefig(filename)
    return figure


def CL(P, alpha, dphr): return dyn.get_aero_coefs(1, alpha, 0, dphr, P)[0]

def plot_CL(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    dphrs = np.deg2rad(np.linspace(20, -30, 3))
    figure = ut.prepare_fig(None, f'Coefficient de portance {P.name}')
    for dphr in dphrs:
        plt.plot(np.rad2deg(alphas), CL(P, alphas, dphr))
    ut.decorate(plt.gca(), u'Coefficient de Portance {}'.format(P.name), r'$\alpha$ en degres', '$C_L$',
                ['$\delta _{{PHR}} =  ${:.1f}'.format(np.rad2deg(dphr)) for dphr in dphrs])
    ut.savefig(filename)

def Cm(P, alpha): return P.Cm0 -P.ms*P.CLa*(alpha-P.a0)

def plot_Cm(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    mss = [-0.1, 0., 0.2, 1.]
    figure = ut.prepare_fig(None, f'Coefficient de moment {P.name}')
    for ms in mss:
        P.set_mass_and_static_margin(0.5, ms)
        plt.plot(np.rad2deg(alphas), Cm(P, alphas))
    ut.decorate(plt.gca(), u'Coefficient de moment {}'.format(P.name), r'$\alpha$ en degres', '$C_m$',
                ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    ut.savefig(filename)


#Question 4

def dphre(P,alpha): return (P.Cm0 +Cm(P, alpha)*(alpha-P.a0))/P.Cmd

def plot_dphre_ms(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    mss=[-0.1,0.,0.2,1]
    figure = ut.prepare_fig(None, f"Angle d'incidence de l'empennage {P.name}")
    for ms in mss:
        P.set_mass_and_static_margin(0.5,ms)
        plt.plot(np.rad2deg(alphas),np.rad2deg(dphre(P,alphas)))
    ut.decorate(plt.gca(), u"Angle d'incidence de l'empennage {}".format(P.name), r'$\alpha$ en degres', '$\delta_{phre}$',
            ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    ut.savefig(filename)
    
#Question 5
def plot_CLe_ae(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    mss=[0.2, 1.]
    figure = ut.prepare_fig(None, f"Coefficient de portance équilibrée {P.name}")
    for ms in mss:
        CLs = []
        for alpha in alphas:
            P.set_mass_and_static_margin(0.5,ms)
            dphreq = dphre(P,alpha)
            CLs.append(CL(P, alpha, dphreq))
        plt.plot(np.rad2deg(alphas),CLs)
    ut.decorate(plt.gca(), u"Coefficient de portance équilibrée {}".format(P.name), r'$\alpha_{e}$ en degres', '$CL_{e}$',
            ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    ut.savefig(filename)


#Question 6
def plot_CL_equilibree(P, filename=None):
    figure = ut.prepare_fig(None, f'Coefficient de portance {P.name}')
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    mss = [0.2, 1.]
    for ms in mss:
        CLs = []
        CDs = []
        for alpha in alphas:
            P.set_mass_and_static_margin(0.5,ms)
            dphr = dphre(P,alpha)
            CLs.append(CL(P, alpha, dphr))
            CDs.append(dyn.get_aero_coefs(1,alpha,0,dphr,P)[1])
        plt.plot(CDs, CLs)
   
    ut.decorate(plt.gca(), u'Coefficient de Portance équilibré {}'.format(P.name), '$C_D$', '$C_L$',
                ['$ms =  ${:.1f}'.format(ms) for ms in mss])
    ut.savefig(filename)
    plt.show()




def seance_1(ac=dyn.Param_737_800()):
    plot_thrust(ac, f'../plots/{ac.get_name()}_thrust.png')
    plot_CL(ac, f'../plots/{ac.get_name()}_CL.png')
    plot_Cm(ac, f'../plots/{ac.get_name()}_Cm.png')
    plot_dphre_ms(ac, f'../plots/{ac.get_name()}_dphre_ms.png')
    plot_CLe_ae(ac, f'../plots/{ac.get_name()}_Cle.png')
    plot_CL_equilibree(ac, f'../plots/{ac.get_name()}_ms.png')

if __name__ == "__main__":
    if 'all' in sys.argv:
        for t in dyn.all_ac_types:
            seance_1(t())
    else:
        P = dyn.Param_737_800()#use_improved_induced_drag = False, use_stall = False)
        seance_1(P)
        plt.show()