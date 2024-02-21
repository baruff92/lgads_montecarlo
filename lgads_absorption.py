import numpy as np
#import matplotlib
#import tkinter

#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.integrate import quad
from scipy import optimize
from scipy import special
from datetime import datetime
import re

W1 = {'e_gain': 1,
      'h_gain': 0.253,
      'th_impl': 112,  # nm
      'depth_gl': 670,  # nm
      'noise': 24 # e-  
      }

W9 = {'e_gain': 1,
      'h_gain': 0.377,
      'th_impl': 107,  # nm
      'depth_gl': 299,  # nm 
      'noise': 25 # e-  
      }

W13 = {'e_gain': 1,
      'h_gain': 0.48,
      'th_impl': 61,  # nm
      'depth_gl': 263,  # nm 
      'noise': 33 # e-  
      }

W17=W1

def lgads_mult():
    print('Calculations for Eiger paper')

    files =     [
    'LGADs_absorption_200eV.txt',
#    'LGADs_absorption_250eV.txt',
    'LGADs_absorption_300eV.txt',
#    'LGADs_absorption_350eV.txt',
    'LGADs_absorption_400eV.txt',
#    'LGADs_absorption_450eV.txt',
    'LGADs_absorption_500eV.txt',
#    'LGADs_absorption_550eV.txt',
    'LGADs_absorption_600eV.txt',
#    'LGADs_absorption_650eV.txt',
    'LGADs_absorption_700eV.txt',
#    'LGADs_absorption_750eV.txt',
    'LGADs_absorption_800eV.txt',
#    'LGADs_absorption_850eV.txt',
    'LGADs_absorption_900eV.txt',
#    'LGADs_absorption_950eV.txt',
    'LGADs_absorption_1000eV.txt',
#    'LGADs_absorption_2000eV.txt',
#    'LGADs_absorption_3000eV.txt',
]
    fig1, sub1 = plt.subplots()        
    QE_plot = np.zeros(len(files))
    energy_plot = np.zeros(len(files))

    figW1, subW1 = plt.subplots() 
    figW9, subW9 = plt.subplots() 
    figW13, subW13 = plt.subplots() 

    subW1.set(xlabel='Energy (eV)', ylabel='Counts')
    subW9.set(xlabel='Energy (eV)', ylabel='Counts')
    subW13.set(xlabel='Energy (eV)', ylabel='Counts')       

    for jj,fn in enumerate(files):
        energy_plot[jj] = int(re.search('_absorption_(.*)eV',fn).group(1))
        print('Photon Energy:', energy_plot[jj], 'eV')
        eventID, posZ, edep = get_simulated_data('data/'+fn)
        #print('Min hit position:',np.min(posZ), 'nm - Max:', np.max(posZ), 'nm')
        if energy_plot[jj] %100 == 0:
            n,b,h = sub1.hist(posZ,500, label=str(energy_plot[jj])+'eV', histtype='step', )
        tot_photons = eventID[-1]
        print('Total photons:', tot_photons)
        recorded_photons = len(eventID)
        print('Recorded photons:', recorded_photons)
        QE_plot[jj] = recorded_photons/tot_photons
        print('QE:', QE_plot[jj])
        th_passivation = np.min(posZ)
        print('Passivation:', th_passivation)

        mult_event_W1 = [np.random.normal(lgad_multiplication(x-th_passivation, e*1000, W1), W1['noise']*3.6,1) for x,e in zip(posZ,edep)]
        nm,bm,hm = subW1.hist(mult_event_W1,500, label=str(energy_plot[jj])+'eV', histtype='step', )

        # mult_event_W9 = [lgad_multiplication(x-th_passivation, e*1000, W9) for x,e in zip(posZ,edep)]
        # nm,bm,hm = subW9.hist(mult_event_W9,500, label=str(energy_plot[jj])+'eV', histtype='step', )

        # mult_event_W13 = [lgad_multiplication(x-th_passivation, e*1000, W13) for x,e in zip(posZ,edep)]
        # nm,bm,hm = subW13.hist(mult_event_W13,500, label=str(energy_plot[jj])+'eV', histtype='step', )

    xplot = np.arange(0,100000,10)
    sub1.plot(xplot, [lgad_multiplication(x-th_passivation, 1000, W1) for x in xplot], label='W1')
    sub1.plot(xplot, [lgad_multiplication(x-th_passivation, 1000, W9) for x in xplot], label='W9')
    sub1.plot(xplot, [lgad_multiplication(x-th_passivation, 1000, W13) for x in xplot], label='W13')
    sub1.legend(frameon=False)
    sub1.plot([0,0, 275000,275000],[0,100000,100000,0], color='orange')
    sub1.set(xlabel='Depth (nm)', ylabel='Counts', ylim=[0,3000],xlim=[-1000,20000])
    fig1.show()    

    fig2, sub2 = plt.subplots()    
    sub2.plot(energy_plot,QE_plot, marker= '.',label='QE')    
    sub2.legend(frameon=False)
    sub2.grid()
    sub2.set(xlabel='Energy (eV)', ylabel='QE')
    fig2.show()    

    subW1.legend(frameon=False)
    subW9.legend(frameon=False)
    subW13.legend(frameon=False)

    figW1.show()
    figW9.show()
    figW13.show()


def get_simulated_data(txt_fname):
    print('Getting data from:', txt_fname)
    data = np.loadtxt(txt_fname)
    print(np.shape(data))
    return data[:,0], data[:,1], data[:,2]

def write_simulated_data_infile(root_fname):
    import ROOT
    print('Getting data from:', root_fname)
    fout = open(root_fname.replace('.root','.txt'),'w')

    dataFile = ROOT.TFile.Open(root_fname,'READ')
    hitsTree = dataFile.Get("Hits")
    entries = hitsTree.GetEntries()
    print('Number of hits:', entries)

    for j,x in enumerate(hitsTree):
        print(x.eventID,-x.posZ*1e6, x.edep*1000, file=fout)
        if j==0 : print(x.eventID,-x.posZ*1e6, x.edep*1000)

    #posZ = np.array([-x.posZ*1e6 for x in hitsTree]) # mm -> nm
    #edep = np.array([x.edep for x in hitsTree]) # MeV
    #return posZ, edep

def convert_all():
    files=[
    'LGADs_absorption_1000eV.root',
    'LGADs_absorption_2000eV.root',
    'LGADs_absorption_3000eV.root',
    'LGADs_absorption_200eV.root',
    'LGADs_absorption_250eV.root',
    'LGADs_absorption_300eV.root',
    'LGADs_absorption_350eV.root',
    'LGADs_absorption_400eV.root',
    'LGADs_absorption_450eV.root',
    'LGADs_absorption_500eV.root',
    'LGADs_absorption_550eV.root',
    'LGADs_absorption_600eV.root',
    'LGADs_absorption_650eV.root',
    'LGADs_absorption_700eV.root',
    'LGADs_absorption_750eV.root',
    'LGADs_absorption_800eV.root',
    'LGADs_absorption_850eV.root',
    'LGADs_absorption_900eV.root',
    'LGADs_absorption_950eV.root',
    ]        
    for f in files:
        write_simulated_data_infile('data/'+f)


def lgad_multiplication(x, e, w):
    if x < 0: return 0
    if x >= 0 and x<= w['th_impl']: return e*w['h_gain']
    if x > w['th_impl'] and x < w['depth_gl']: 
        m = e*(w['e_gain']-w['h_gain'])/(w['depth_gl']-w['th_impl'])
        q = e*w['h_gain']-m*w['th_impl']
        return x*m + q
    if x>= w['depth_gl']: return e*w['e_gain']

