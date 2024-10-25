import numpy as np
#import matplotlib
#import tkinter

#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.integrate import dblquad
from scipy.integrate import quad
from scipy import optimize
from scipy import special
from datetime import datetime
import re
import time
import multiprocessing
from itertools import repeat
from lmfit import Model,Parameters,Minimizer, report_fit
from lmfit.models import *
import copy
from scipy.optimize import curve_fit
from matplotlib import rcParams
rcParams['font.sans-serif'] = "Times"
rcParams['font.family'] = "Arial"
plt.rc('axes', unicode_minus=False)

W1 = {'e_gain': 1,
      'h_gain': 0.253,
      'h_gain_err': 0.003,
      'th_impl': 112,  # nm
      'depth_gl': 670,  # nm
      'depth_gl_err': 12,  # nm
      'noise': 23, # e-
      'charge_cloud': 7.36,
      'charge_cloud_err': 0.15  
      }

W9 = {'e_gain': 1,
      'h_gain': 0.377,
      'h_gain_err': 0.003,
      'th_impl': 107,  # nm
      'depth_gl': 299,  # nm 
      'depth_gl_err': 5,  # nm
      'noise': 22.3, # e-  
      'charge_cloud': 4.88,
      'charge_cloud_err': 0.08  
      }

W13 = {'e_gain': 1,
      'h_gain': 0.48,
      'h_gain_err': 0.01,
      'th_impl': 61,  # nm
      'depth_gl': 263,  # nm 
      'depth_gl_err': 18,  # nm
      'noise': 33 # e-  
      }

W17 = {'e_gain': 1,
      'h_gain': 0.253,
      'h_gain_err': 0.003,
      'th_impl': 112,  # nm
      'depth_gl': 670,  # nm
      'depth_gl_err': 12,  # nm
      'noise': 24 # e-  
      }      
      
def lgads_mult():
    print('Calculations for Eiger paper')

    files = [
    # 'LGADs_absorption_200eV.txt',
    # 'LGADs_absorption_250eV.txt',
    # 'LGADs_absorption_300eV.txt',
    # 'LGADs_absorption_350eV.txt',
    # 'LGADs_absorption_400eV.txt',
    # 'LGADs_absorption_450eV.txt',
    'LGADs_absorption_500eV.txt',
    'LGADs_absorption_550eV.txt',
    'LGADs_absorption_600eV.txt',
    'LGADs_absorption_650eV.txt',
    'LGADs_absorption_700eV.txt',
    'LGADs_absorption_750eV.txt',
    'LGADs_absorption_800eV.txt',
    'LGADs_absorption_850eV.txt',
    'LGADs_absorption_900eV.txt',
#    'LGADs_absorption_950eV.txt',
#    'LGADs_absorption_1000eV.txt',
#    'LGADs_absorption_2000eV.txt',
#    'LGADs_absorption_3000eV.txt',
]
    fig1, sub1 = plt.subplots()        
    QE_plot = np.zeros(len(files))
    energy_plot = np.zeros(len(files))

    QE_plotW1 = np.zeros(len(files))
    QE_plotW9 = np.zeros(len(files))
    QE_plotW13 = np.zeros(len(files))

    figW1, subW1 = plt.subplots() 
    figW9, subW9 = plt.subplots() 
    figW13, subW13 = plt.subplots() 

    subW1.set(xlabel='Energy (eV)', ylabel='Counts', title='W1')
    subW9.set(xlabel='Energy (eV)', ylabel='Counts', title='W9')
#    subW13.set(xlabel='Energy (eV)', ylabel='Counts', title='W13')       

    figW9comp, subW9comp = plt.subplots() 
    subW9comp.set(xlabel='Energy (eV)', ylabel='Counts', title='W9')
    
    comparison_files= ['data/W9_vrpre3500_E900eV.txt',
                    'data/W9_vrpre3500_E800eV.txt',
                    'data/W9_vrpre3500_E700eV.txt',
                    'data/W9_vrpre3700_E600eV.txt',
    ]
    plot_c = [50,45,45,40]
    
    W1c = copy.deepcopy(W1)
    W9c = copy.deepcopy(W9)
    W1c['h_gain'] = np.random.normal(W1['h_gain'], 0*W1['h_gain_err'],1)[0]	
    W9c['h_gain'] = np.random.normal(W9['h_gain'], 0*W9['h_gain_err'],1)[0]
    W1c['depth_gl'] = np.random.normal(W1['depth_gl'], 0*W1['depth_gl_err'],1)[0]	
    W9c['depth_gl'] = np.random.normal(W9['depth_gl'], 0*W9['depth_gl_err'],1)[0]	
          
    print('W1:',W1c)  
    print('W9:',W9c)  
        
    for jj,fn in enumerate(files):
        energy_plot[jj] = int(re.search('_absorption_(.*)eV',fn).group(1))
        print('Photon Energy:', energy_plot[jj], 'eV')
        eventID, posZ, edep = get_simulated_data('data/'+fn)
        #print('Min hit position:',np.min(posZ), 'nm - Max:', np.max(posZ), 'nm')
        if energy_plot[jj] %100 == 0:
            n,b,h = sub1.hist(posZ,500, label=str(energy_plot[jj])+'eV', histtype='step', )
        tot_photons = eventID[-1]+1
        #print('Total photons:', tot_photons)
        recorded_photons = len(eventID)
        #print('Recorded photons:', recorded_photons)
        QE_plot[jj] = recorded_photons/tot_photons
        #print('QE:', QE_plot[jj])
        th_passivation = np.min(posZ)
#        print('Passivation:', th_passivation)
        
        QE_plotW1[jj] = np.count_nonzero(posZ > W1c['depth_gl']-th_passivation)/recorded_photons
        QE_plotW9[jj]= np.count_nonzero(posZ > W9c['depth_gl']-th_passivation)/recorded_photons
#        QE_plotW13[jj]= np.count_nonzero(posZ > W13['depth_gl']-th_passivation)/recorded_photons
        
        mult_event_W1 = [lgad_multiplication(x-th_passivation, e*1000, W1c, 0*W1c['noise']*3.6) for x,e in zip(posZ,edep)]
        mult_event_W9 = [lgad_multiplication(x-th_passivation, e*1000, W9c, 0*W9c['noise']*3.6) for x,e in zip(posZ,edep)]
#        mult_event_W13 = [lgad_multiplication(x-th_passivation, e*1000, W13, 0*W13['noise']*3.6) 
#                                                        for x,e in zip(posZ,edep)]

        foutw1 = open('data/mult_spectr_W1_E'+f'{energy_plot[jj]:0.0f}'+'eV.txt','w')
        for el in mult_event_W1: print(el, file=foutw1)
        foutw9 = open('data/mult_spectr_W9_E'+f'{energy_plot[jj]:0.0f}'+'eV.txt','w')
        for el in mult_event_W9: print(el, file=foutw9)
#        foutw13 = open('data/mult_spectr_W13_E'+f'{energy_plot[jj]:0.0f}'+'eV.txt','w')
#        for el in mult_event_W13: print(el, file=foutw13)
        
        if energy_plot[jj] %100 == 0:
            nm,bm,hm = subW9.hist(mult_event_W9,200, label=str(energy_plot[jj])+'eV', histtype='step', )
            nm,bm,hm = subW1.hist(mult_event_W1,200, label=str(energy_plot[jj])+'eV', histtype='step', )
#            nm,bm,hm = subW13.hist(mult_event_W13,200, label=str(energy_plot[jj])+'eV', histtype='step', )

            if energy_plot[jj]  > 560:
                nm,bm,hm = subW9comp.hist(np.random.normal(mult_event_W9, W9['noise']*3.6),200, label=str(energy_plot[jj])+'eV', histtype='step', )

                for comp, ic in zip(comparison_files,plot_c):
                    if 'E'+str(energy_plot[jj]).replace('.0','')+'eV' in comp:
                        data = np.loadtxt(comp)
                        norm = np.max(nm)/np.max(data[:,1])
                        subW9comp.plot(data[ic:,0],data[ic:,1]*norm, '--', label=str(energy_plot[jj])+'eV data')

    xplot = np.arange(0,100000,10)
    sub1.plot(xplot, [lgad_multiplication(x-th_passivation, 1000, W1c) for x in xplot], label='W1')
    sub1.plot(xplot, [lgad_multiplication(x-th_passivation, 1000, W9c) for x in xplot], label='W9')
#    sub1.plot(xplot, [lgad_multiplication(x-th_passivation, 1000, W13) for x in xplot], label='W13')
    sub1.legend(frameon=False)
    sub1.plot([0,0, 275000,275000],[0,100000,100000,0], color='orange')
    sub1.set(xlabel='Depth (nm)', ylabel='Counts', ylim=[0,3000],xlim=[-1000,20000])
    fig1.show()    

    fig2, sub2 = plt.subplots()    
    sub2.plot(energy_plot,QE_plot, marker= '.',label='QE')    
    sub2.plot(energy_plot,QE_plotW1, marker= '.',label='QE W1')    
    sub2.plot(energy_plot,QE_plotW9, marker= '.',label='QE W9')    
#    sub2.plot(energy_plot,QE_plotW13, marker= '.',label='QE W13')    

    sub2.legend(frameon=False)
    sub2.grid()
    sub2.set(xlabel='Energy (eV)', ylabel='QE')
    fig2.show()

    subW1.legend(frameon=False)
    subW9.legend(frameon=False)
#    subW13.legend(frameon=False)

    figW1.show()
    figW9.show()
#    figW13.show()
    subW9comp.legend()
    figW9comp.show()
    
    print('Multiplied W1', QE_plotW1)
    print('Multiplied W9', QE_plotW9)
#    print('Multiplied W13', QE_plotW13)

def charge_sharing_spectrum():
    energies = np.arange(500,901,50)
    wafers = ['W1','W9',]#, 'W1', 'W13','W17']
    w_dics = [W1, W9]#,W1,W13,W17]
    tot_photons = 1e5 # number of simulated photons  
    QE=np.zeros((len(wafers),len(energies)))  
    QE_mult=np.zeros((len(wafers),len(energies)))  
        
    fig1, sub1 = plt.subplots() # weighting field
    pp = 75 # um

    # chcloud = np.loadtxt('data/charge_collection4.68um_pp75um.txt')
    # chcloud = np.loadtxt('data/charge_collection6.00um_pp75um.txt')
    cloud = '6.00'

    # im = sub1.imshow(chcloud, extent=(0,pp,0,pp))
    # sub1.plot([0,pp,pp,0,0],[0,0,pp,pp,0],'-', color='black')
    # sub1.set(xlabel='x(um)',ylabel='y(um)', title='Charge collection')
    # fig1.colorbar(im)
    # fig1.show()
    
    comparison_files= [
                # 'data/W1_vrpre3400_E900eV.txt',
#                'data/W17_vrpre2800_E900eV.txt',
               'data/W1_vrpre3400_E900eV.txt',
                # 'data/W1_vrpre3400_E850eV.txt',
                # 'data/W1_vrpre3400_E800eV.txt',
#                 'data/W1_vrpre3400_E600eV.txt',
#                'data/W1_vrpre3400_E600eV.txt',
                # 'data/W9_vrpre3500_E800eV.txt',
                # 'data/W9_vrpre3500_E850eV.txt',
                # 'data/W9_vrpre3500_E700eV.txt',
                # 'data/W9_vrpre3700_E600eV.txt',
    ]


    plot_c = [50,45,45,40]
    plot_c = [46,45,42,40] #w9
#    plot_c = [50,48,47,42] #w1

    fig5, sub5 = plt.subplots()
    sub5.set(xlabel='Energy (eV)', ylabel='Counts')           

    fig7, sub7 = plt.subplots()
    sub7.set(xlabel='Energy (eV)', ylabel='Counts', title='Comparison W1-W9')           

    model_g = GaussianModel()

    for iw,(w,ws) in enumerate(zip(wafers,w_dics)):

        cl_dim = 7
        cl_dim_r = round(cl_dim ,1)
        chcloud = np.loadtxt('data/charge_collection/charge_collection'+f'{cl_dim_r:.2f}'+'um_pp75um.txt')

        print(w,ws)
        fig2, sub2 = plt.subplots()
        sub2.set(xlabel='Energy (eV)', ylabel='Counts', title=w+' No charge-sharing')           
        fig3, sub3 = plt.subplots()
        sub3.set(xlabel='Energy (eV)', ylabel='Counts', title=w+' With charge-sharing')           
        sub3.plot([ws['noise']*3.6*5,ws['noise']*3.6*5],[0,2000], ':')
        fig4, sub4 = plt.subplots()
        sub4.set(xlabel='Energy (eV)', ylabel='Counts', title=w+' With and without charge-sharing')           

        for ie,e in enumerate(energies):
            pool = multiprocessing.Pool()
            txt_fname = 'data/mult_spectr_'+w+'_E'+f'{e:0.0f}'+'eV.txt'
            if w=='W17': txt_fname = 'data/mult_spectr_W1_E'+f'{e:0.0f}'+'eV.txt'
            data = np.loadtxt(txt_fname)
            data_noise_nosh = np.random.normal(data, ws['noise']*3.6)
            QE[iw,ie] = np.shape(data)[0]/tot_photons

            x_c = np.random.uniform(0,pp, size=len(data))
            y_c = np.random.uniform(0,pp, size=len(data))

            #data_chsh = np.array([integral_gauss2d(x,y,sigma_cloud,d,pp) for x,y,d in zip(x_c,y_c,data)] )
            data_chsh = np.array(list(pool.starmap(get_charge_collected, 
                                        zip(x_c,y_c,data,repeat(chcloud),repeat(pp)))))
            data_noise = np.random.normal(data_chsh, ws['noise']*3.6)
            pool.close
            
            hist, bin = np.histogram(data_noise, bins=150, range=[0,1500])
			
            # foutname = 'sim_data/'+w+'_E'+str(e)+'eV_chcl'+ cloud +'um_4.txt'
            # fout = open(foutname,'w')
            # for ix,iy in zip(bin, hist):
            #     print(ix,iy, file=fout)

            QE_mult[iw,ie] = np.count_nonzero(data_noise > ws['noise']*5*3.6)/np.shape(data)[0]

            if e %100 == 0:    
                nm,bm,hm = sub2.hist(data_noise_nosh,200, label=str(e)+'eV', histtype='step')
                nm,bm,hm = sub3.hist(data_noise,200, label=str(e)+'eV', histtype='step')

            if e == 900:    
                nm,bm,hm = sub4.hist(data_noise_nosh,200, label=str(e)+'eV - NO charge sharing', histtype='step')
                nm,bm,hm = sub4.hist(data_noise,200, label=str(e)+'eV - with charge sharing', histtype='step')
                nm,bm,hm = sub7.hist(data_noise,200, label=str(e)+'eV - '+w, histtype='step')
               
            if w=='W1':
                for comp, ic in zip(comparison_files,plot_c):
                    if 'E'+str(e).replace('.0','')+'eV' in comp and w in comp:
                        minf = 67
                        maxf = 85
                        minf1 = 135
                        maxf1 = 170  
                        data = np.loadtxt(comp)
                        norm = np.max(nm[100:])/np.max(data[:,1])
                        sub5.plot(data[ic:,0],data[ic:,1]*norm, '-', label=str(e)+'eV data')
                        nd,bd,hd = sub5.hist(data_noise,200, label=str(e)+'eV sim', histtype='step')
                        params_g = model_g.make_params(center=900, amplitude=2000, sigma=20)
                        result_g = model_g.fit(data[minf:maxf,1]*norm, params_g, x=data[minf:maxf,0])
                        sub5.plot(data[minf:maxf,0], result_g.best_fit, '-', label='fit data')
                        print('Data fit:', result_g.params.valuesdict())

                        params_g = model_g.make_params(center=900, amplitude=2000, sigma=20)
                        result_g = model_g.fit(nd[minf1:maxf1], params_g, x=bd[minf1:maxf1])
                        sub5.plot(bd[minf1:maxf1], result_g.best_fit, '-', label='fit simu')
                        print('Simu fit:', result_g.params.valuesdict())

    for comp, ic in zip(comparison_files,plot_c):
        data = np.loadtxt(comp)
        norm = np.max(nm)/np.max(data[:,1])
        sub7.plot(data[ic:,0],data[ic:,1]*norm, '--', label=str(e)+'eV data')

        sub2.legend()    
        fig2.show()
        sub3.legend()    
        fig3.show()
        sub4.legend()    
        fig4.show()
        sub5.legend()    
        fig5.show()
        sub7.legend()    
        fig7.show()

    fig6, sub6 = plt.subplots()   
    sub6.set(xlabel='Energy (eV)', ylabel='Efficiency', title='Efficiency')            
    for iw,(w,ws) in enumerate(zip(wafers,w_dics)):
        sub6.plot(energies,QE[iw,:], marker= '.',label='QE - '+w)  
        sub6.plot(energies,QE_mult[iw,:], marker= '.',label='QE mult - '+w)  
        
    #print(QE)        
    #print(QE_mult)

    sub6.legend()    
    fig6.show()

    return QE_mult
       
def calculate_charge_collection(sigma_cloud = 4.88):
#    sigma_cloud = 4.68 #um
#    sigma_cloud = 6 #um
#    sigma_cloud = 6.5 #um
    pp = 75 # um
    Evgridx = np.arange(0,pp+.1,1)
    Evgridy = np.arange(0,pp+.1,1)    
    X, Y = np.meshgrid(Evgridx, Evgridy)
        
    integral_gauss2d_v = np.vectorize(integral_gauss2d)
    chcloud = integral_gauss2d_v(X,Y,sigma_cloud,1,pp)

    fig2, sub2 = plt.subplots() # weighting field    
    im = sub2.imshow(chcloud, extent=(0,pp,0,pp), vmin=0, vmax=1 ,cmap='inferno')
    sub2.plot([0,pp,pp,0,0],[0,0,pp,pp,0],'-', color='black')
    sub2.set(xlabel='x ($\\mu$m)',ylabel='y ($\\mu$m)')
    sub2.tick_params(direction='in')
    fig2.colorbar(im, label='Collected charge fraction')
    fig2.show()

    fout = open('data/charge_collection/charge_collection'+f'{sigma_cloud:0.2f}'+'um_pp'+f'{pp:0.0f}'+'um'+'.txt','w')
    for el in chcloud: print(*el, file=fout)

def gauss2d(x,y,x0,y0,sigma,I):
    return I/(2*np.pi*np.power(sigma,2))*np.exp(-0.5*np.power(x-x0,2)/np.power(sigma,2)-0.5*np.power(y-y0,2)/np.power(sigma,2))
    
def integral_gauss2d(x0,y0,sigma,energy,pp):
    x,err = dblquad(gauss2d, 0, pp, 0, pp, args=(x0,y0,sigma,energy),epsabs=0.01)    
    return x

def get_charge_collected(x,y,e,coll_map,pp):
    #print(x,y,e,pp)
    return e*coll_map[int(x/pp*np.shape(coll_map)[0]),int(y/pp*np.shape(coll_map)[1])]
    
def get_simulated_data(txt_fname):
    # print('Getting data from:', txt_fname)
    data = np.loadtxt(txt_fname)
    #print(np.shape(data))
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

def get_QE_data_antonio():
    import ROOT
    fileroot = '/mnt/sls_det_storage/eiger_data/lgad/Filippo_analysis/lgads_sim/lgads_montecarlo/plots/qe/qe_vs_energy.root'
    
    dataFile = ROOT.TFile.Open(fileroot,'READ')
    canvas = ["qe_vs_energy_W1_pin","qe_vs_energy_W9_pin"]

    for c in canvas:
        print(c)
        can = dataFile.Get(c)
        #print(can.ls())
        graph = can.FindObject("Graph")
        N = graph.GetN()
        print('Points', N)
        for n in range(N):
            print(graph.GetPointX(n),graph.GetPointY(n),graph.GetErrorY(n)) 

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


def lgad_multiplication(x, e, w, noise=0):
    if x < 0: return 0
    if x >= 0 and x<= w['th_impl']: return np.random.normal(e*w['h_gain'], noise,1)[0]
    if x > w['th_impl'] and x < w['depth_gl']: 
        m = e*(w['e_gain']-w['h_gain'])/(w['depth_gl']-w['th_impl'])
        q = e*w['h_gain']-m*w['th_impl']
        return np.random.normal(x*m + q, noise,1)[0]
    if x>= w['depth_gl']: return np.random.normal(e*w['e_gain'], noise,1)[0]

def lgad_multiplication_noiseless(x, e, w):
    if x < 0: return 0
    if x >= 0 and x<= w['th_impl']: return e*w['h_gain']
    if x > w['th_impl'] and x < w['depth_gl']: 
        m = e*(w['e_gain']-w['h_gain'])/(w['depth_gl']-w['th_impl'])
        q = e*w['h_gain']-m*w['th_impl']
        return x*m + q
    if x>= w['depth_gl']: return e*w['e_gain']

def plot_SNR():
	print('This is the SNR of LGAD-Eiger')
	
	# W9 vrpre 3500
	snr_W9 = np.array([[900,11.778,0.682],
			   [850,10.779,0.608],
			   [800,10.047,0.656],
			   [750,9.084,0.729],
			   [700,8.346,0.747],
			   [650,7.273,0.745],
			   [600,6.010,1.067],
			   [550,4.914,0.901],])				

	# W1 vrpre 3400
	snr_W1 = np.array([
			[900,10.812,0.763],
			[850,10.473,0.669],
			[800,10.023,0.603],
			[750,9.184,0.567],
			[700,8.514,0.489],
			[650,7.843,0.486],
			[600,7.044,0.540],
			[550,5.768,0.669],
	])				

	# W13 vrpre 3400
	snr_W13 = np.array([[900,8.436],
				[850,7.902],
				[800,7.110],
				[750,6.233]])				

	fig1, sub1 = plt.subplots(2,1) 
	fig1.subplots_adjust(hspace=0)
	sub1[1].set(xlabel='Energy (eV)')
	sub1[0].set_ylabel('Signal-to-noise Ratio')

	#w9
	sub1[0].errorbar(snr_W9[:,0]-1,snr_W9[:,1], yerr=snr_W9[:,2], 
		marker = 'o',mfc='none', color='black', linestyle = '--',capsize=3, capthick=1, label='Shallow')
#	sub1.plot(np.arange(0,2000,100), par_W9[0]*np.arange(0,2000,100)+par_W9[1], '--')

	#w1
	sub1[0].errorbar(snr_W1[:,0]+1,snr_W1[:,1], yerr=snr_W1[:,2], 
		marker = 's', mfc='none', color='red', linestyle = '--',capsize=3, capthick=1, label='Standard')
#	sub1.plot(np.arange(0,2000,100), par_W1[0]*np.arange(0,2000,100)+par_W1[1], '--')

	#w13
#	sub1.plot(snr_W13[:,0],snr_W13[:,1], marker = 'o', linestyle = 'none', label='Ultra-shallow' )
#	sub1.plot(np.arange(0,2000,100), par_W13[0]*np.arange(0,2000,100)+par_W13[1], '--')
	
	#en, measuredQE, err signal>5sigma, err, tot_eff, err, photons absorbed after gain layer
	eff_W1 = np.array([ 
	[200,0.4324480,0.0161546,0.0000066,0.0000128,0.0000029,0.0000055],
	[250,0.5620353,0.0111879,0.0000571,0.0000328,0.0000321,0.0000184],
	[300,0.6133537,0.0100897,0.0008740,0.0001202,0.0005360,0.0000743],
	[350,0.6875259,0.0351471,0.0072646,0.0004225,0.0049946,0.0003868],
	[400,0.7231155,0.0150489,0.0330144,0.0012195,0.0238732,0.0010121],
	[450,0.7600932,0.0117162,0.0948216,0.0024360,0.0720732,0.0021593],
	[500,0.8006333,0.0134015,0.1943153,0.0036701,0.1555753,0.0039263],
	[550,0.7319474,0.0121542,0.3095837,0.0044671,0.2265990,0.0049849],
	[600,0.7845862,0.0130894,0.4249784,0.0044976,0.3334322,0.0065876],
	[650,0.8203867,0.0137586,0.5255493,0.0041211,0.4311537,0.0079822],
	[700,0.8554985,0.0128204,0.6063890,0.0036887,0.5187648,0.0083902],
	[750,0.8648947,0.0131384,0.6777788,0.0030917,0.5862073,0.0092977],
	[800,0.8799206,0.0134775,0.7334904,0.0027396,0.6454133,0.0101753],
	[850,0.8949545,0.0148728,0.7805066,0.0023967,0.6985179,0.0118048],
	[900,0.9167479,0.0151788,0.8207330,0.0018965,0.7524052,0.0125785],
			])
	
	eff_W9 = np.array([
[200,0.489865363,0.011013155,4.4951059348284E-05,3.59046467721366E-05,2.20199670048817E-05,1.75954084477984E-05],
[250,0.588656204,0.012524678,0.00060453367624,0.000109502009654	,0.000355862499046,6.49022069233305E-05	],
[300,0.655403506,0.010526522,0.005140703840945,0.000303214396493,0.003369235320663,0.000205963652034],
[350,0.700047389,0.035989005,0.024731934271546,0.000676349139337,0.017313526011716,0.001008175714249],
[400,0.743308963,0.015994755,0.078165820844212,0.001285225836222,0.058101355233755,0.001573449720664],
[450,0.72288984,0.010942573,0.177213070303983,0.002088177764691,0.128105528037955,0.002457443110394],
[500,0.7747794,0.012306435,0.311416398391522,0.002543349177612,0.241279010295945,0.004309349473852],
[550,0.695852713,0.010043065,0.449057249214052,0.002657826399318,0.312477705157915,0.004874400982644],
[600,0.746760343,0.01182466,0.567102197917095,0.002306961669087,0.423489431832624,0.006923545875507],
[650,0.779146386,0.012495118,0.655961225149793,0.002250445661574,0.511089817931594,0.008381768918433],
[700,0.807935326,0.013206563,0.723109328761072,0.001935629639503,0.584225571266218,0.009676990100162],
[750,0.821818386,0.012980527,0.777118142276171,0.001586375580213,0.638649977416721,0.010171301118449],
[800,0.848448683,0.012916812,0.818192518609862,0.001412663907154,0.694194364854991,0.010636187217132],
[850,0.847307341,0.013584308,0.851345370693691,0.001158786454194,0.721351182315131,0.011606541720937],
[900,0.858738425,0.013589013,0.881062503719224,0.001056870584627,0.756602226770403,0.012007119144909],
			])

	QEgeant = np.array([[200,0.4036],
			[250,0.5645],
			[300,0.6835],
			[350,0.7680],
			[400,0.8231],
			[450,0.7779],
			[500,0.8222],
			[550,0.7865],
			[600,0.8197],
			[650,0.8476],
			[700,0.8717],
			[750,0.8905],
			[800,0.9081],
			[850,0.9211],
			[900,0.9321],
			])

	sub1[1].errorbar(eff_W1[7:,0],eff_W1[7:,1]*100,yerr=eff_W1[7:,2]*100,marker = '.',mfc='none', color='grey', linestyle = ':',capsize=3, capthick=1, label='QE')
	sub1[1].errorbar(eff_W9[7:,0],eff_W9[7:,5]*100,yerr=eff_W9[7:,6]*100,marker = 'o',mfc='none', color='black', linestyle = '--',capsize=3, capthick=1, label='Shallow')
	sub1[1].errorbar(eff_W1[7:,0],eff_W1[7:,5]*100,yerr=eff_W1[7:,6]*100,marker = 's',mfc='none', color='red', linestyle = '--',capsize=3, capthick=1, label='Standard')
	sub1[0].tick_params(bottom=True, top=True, left=True, right=True, direction="in")
	sub1[1].tick_params(bottom=True, top=False, left=True, right=True, direction="in", labelleft=False)
	
	ax1 = sub1[1].twinx()
	sub1[1].set_ylim([0.1*100,100])
	ax1.set_ylim([0.1*100,100])
	ax1.tick_params(direction="in")
	ax1.set(ylabel='Efficiency (%)')

	sub1[0].legend(frameon=False)
	sub1[1].legend(frameon=False, loc='lower right')
#	sub1.set_xlim([450,1000])
	fig1.show()

def noise_vs_gain(x, g0,n0):
	return g0/x*n0

def plot_noise_vs_gain():
	noise = np.array([
['W0', 0.238,	0.019,	80.495, 4.219, 'Conventional Sensor (G=1)', 'gray','.'],
['W13',	0.562,	0.160,	33.686, 6.971, 'ultra-Shallow iLGAD (G=2.4)' , 'green','v'],
['W9', 1.165,	0.120,	22.987, 1.415, 'Shallow iLGAD (G=4.9)','black', 's'],
['W1', 1.505,	0.121,	23.032, 1.594, 'Standard iLGAD (G=6.3)', 'red', 'o'],
['W17',	1.354,	0.070,	23.957,1.816, 'Standard iLGAD (G=11)', 'blue','x']]	
	)
	
	fig2, sub2 = plt.subplots() 
	sub2.set_xlabel('Calibration Gain (mV/eV)')
	sub2.set_ylabel('Noise (e-)')
	
	for i, x in enumerate(noise):
		sub2.errorbar(float(x[1]),float(x[3]), xerr=float(x[2]), yerr=float(x[4]), 
	marker = x[7], mfc='none', color=x[6], linestyle = 'none', capsize=3, capthick=1, label=x[5])

	param, cov = curve_fit(noise_vs_gain,noise[:,1].astype(float),noise[:,3].astype(float),
		p0=[float(noise[0,1]),float(noise[0,3])], sigma=noise[:,4].astype(float), absolute_sigma=True)

	xp = np.arange(0.01,2,0.05)
	sub2.plot(xp, noise_vs_gain(xp, param[0],param[1]), '--', color='gray', label='fit') 
	#sub2.plot(xp, noise_vs_gain(xp, 0.238,80.495), '--')

	sub2.legend(frameon=False)	
	sub2.set(xlim=[0,2], ylim=[0,100])
	fig2.show()
	
	noisered = np.array([
['W13', 2.362, 0.699, 2.550, 0.185, 'ultra-Shallow iLGAD (G=2.4)' , 'green','v'],
['W9', 4.897, 0.636, 3.903, 0.185, 'Shallow iLGAD (G=4.9)','black', 's'],
['W1', 6.325, 0.712, 3.823, 0.217, 'Standard iLGAD (G=6.3)', 'red', 'o'],
['W17', 10.967, 0.823, 5.310, 0.399, 'Standard iLGAD (G=11)', 'blue','x']		
	])
	
	fig1, sub1 = plt.subplots() 
	sub1.set_xlabel('LGAD Gain')
	sub1.set_ylabel('Noise reduction factor')

	for i, x in enumerate(noisered):
		sub1.errorbar(float(x[1]),float(x[3]), xerr=float(x[2]), yerr=float(x[4]), 
	marker = x[7], mfc='none', color=x[6], linestyle = 'none', capsize=3, capthick=1, label=x[5])

	xp = np.arange(0,20,0.1)	
	sub1.plot(xp, xp, '--', color='gray', label='linear reduction')
	
	sub1.legend(frameon=False)	
	sub1.set(xlim=[0,12], ylim=[0,6])
	fig1.show()
	
def plot_noise_red():
	print('This is the gain reduction of LGADs')
	w13 = np.array([[2.3618, 0.6988,2.5504,0.2806],plt.rc('axes', unicode_minus=False)
		[2.6147, 0.7073,2.5497,0.2769]])
		
	w9 = np.array([[4.00349,0.73209,4.40278,0.40519],
		[4.57921,0.60255,3.85069,0.25016],
		[4.89665,0.63632,3.90261,0.18545],
		[4.55955,0.59887,3.98145,0.17937]])

	w1 = np.array([[5.46163,0.82645,4.57743,0.51676],
		[6.21893,0.67671,3.89333,0.17343],
		[6.32463,0.71182,3.82325,0.21671],
		[5.61013,0.78452,3.43002,0.18161]])
		
	w17 = np.array([[11.41817,1.08154,5.41726,0.36636],
		[10.97103,0.82752,4.97556,0.34575],
		[10.96706,0.82312,5.31018,0.39868]])

	fig1, sub1 = plt.subplots() 
	sub1.set_xlabel('LGAD multiplication factor')
	sub1.set_ylabel('Noise reduction')

	sub1.errorbar(w13[:,0],w13[:,2], xerr=w13[:,1], yerr=w13[:,3], 
	marker = 'v',mfc='none', color='green', linestyle = 'none', capsize=3, capthick=1, label='ultra-Shallow')

	sub1.errorbar(w9[:,0],w9[:,2], xerr=w9[:,1], yerr=w9[:,3], 
	marker = 's',mfc='none', color='black', linestyle = 'none', capsize=3, capthick=1, label='Shallow')

	sub1.errorbar(w1[:,0],w1[:,2], xerr=w1[:,1], yerr=w1[:,3], 
	marker = 'o',mfc='none', color='red', linestyle = 'none', capsize=3, capthick=1, label='Standard')

	sub1.errorbar(w17[:,0],w17[:,2], xerr=w17[:,1], yerr=w17[:,3], 
	marker = '^',mfc='none', color='blue', linestyle = 'none', capsize=3, capthick=1, label='Standard - high gain')
	
	sub1.plot([0,20],[0,20],'--', color='grey', label='linear')
	sub1.legend(frameon=False)	
	sub1.set(xlim=[1,13], ylim=[1,6])
	fig1.show()
	
	fig2, sub2 = plt.subplots() 
	sub2.set_xlabel('LGAD multiplication factor')
	sub2.set_ylabel('Noise (e-)')
	
	vrpre3500 = np.array([[1.000,0.102,81.246,3.073],
		[2.615,0.707,31.865,3.244],
		[4.560,0.599,22.328,0.547],
		[5.610,0.785,23.687,0.878]])

	vrpre3400 = np.array([[1.000,0.079,87.510,3.111],
		[2.362,0.699,34.312,3.573],
		[4.897,0.636,22.423,0.707],
		[6.325,0.712,22.889,1.011]])

	sub2.errorbar(vrpre3500[:,0],vrpre3500[:,2], xerr=vrpre3500[:,1], yerr=vrpre3500[:,3], 
	marker = 's',mfc='none', color='black', linestyle = '-', capsize=3, capthick=1, label='vrpre 3500')

	sub2.errorbar(vrpre3400[:,0],vrpre3400[:,2], xerr=vrpre3400[:,1], yerr=vrpre3400[:,3], 
	marker = 'o',mfc='none', color='red', linestyle = '-', capsize=3, capthick=1, label='vrpre 3400')

	sub2.plot(np.arange(0.1,20,0.1), vrpre3500[0,2]/np.arange(0.1,20,0.1),'--', color='grey', label='ideal')
	
	
#	sub2.plot([0,20],[0,20],'--', color='grey', label='linear')
	sub2.legend(frameon=False)	
	sub2.set(xlim=[0,8], ylim=[0,100])
	fig2.show()
    
def calculate_efficiency_error():
	iterations = 5
	energies = np.arange(500,901,50)

	efficiency = np.zeros((iterations,2,len(energies)))

	for i in range(iterations):
		print('Iteration:', i)
		lgads_mult()
		efficiency[i] = charge_sharing_spectrum()
		plt.close('all')

	fig1, sub1 = plt.subplots() 
	fig2, sub2 = plt.subplots()
	sub1.set(title='W1', xlabel='Energy (eV)', ylabel='Efficiency')
	sub2.set(title='W9', xlabel='Energy (eV)', ylabel='Efficiency')
	for i in range(iterations):
		sub1.plot(energies,efficiency[i,0,:])	
		sub2.plot(energies,efficiency[i,1,:])
	fig1.show()	
	fig2.show()	
	
	model_g = GaussianModel()

	fig3, sub3 = plt.subplots() 
	fig4, sub4 = plt.subplots()
	sub3.set(title='W1', xlabel='Efficiency', ylabel='Counts')
	sub4.set(title='W9', xlabel='Efficiency', ylabel='Counts')
	for i,e in enumerate(energies):
		print(e)
		n1,b1,h1 = sub3.hist(efficiency[:,0,i],200, label=str(e)+'eV', histtype='step', )
		params_g = model_g.make_params(center=b1[np.argmax(n1)], amplitude=np.amax(n1), sigma=np.std(efficiency[:,0,i])) 
		result_g = model_g.fit(n1, params_g, x=(b1[1:] + b1[:-1])/2)
		sub3.plot((b1[1:] + b1[:-1])/2, result_g.best_fit, '-')
		print(result_g.params.valuesdict())
		print('Average:',np.average(efficiency[:,0,i]))
		print('std.dev',np.std(efficiency[:,0,i]))

		n9,b9,h9 = sub4.hist(efficiency[:,1,i],200, label=str(e)+'eV', histtype='step', )
		params_g = model_g.make_params(center=b9[np.argmax(n9)], amplitude=np.amax(n9), sigma=np.std(efficiency[:,1,i])) 
		result_g = model_g.fit(n9, params_g, x=(b9[1:] + b9[:-1])/2)
		sub4.plot((b9[1:] + b9[:-1])/2, result_g.best_fit, '-')
		print(result_g.params.valuesdict())
		print('Average:',np.average(efficiency[:,1,i]))
		print('std.dev',np.std(efficiency[:,1,i]))
		
	fig1.show()	
	fig2.show()	
	fig3.show()	
	fig4.show()
	
def exp(x,a,t):
	return a*np.exp(-x*t)		

def plot_multiplication_factor():
	fig1, sub1 = plt.subplots() 
	sub1.set( xlabel='Depth (nm)', ylabel='Effective Multiplication factor', ylim=[0,10])
	xplot = np.arange(0,1500,10)
	th_passivation = 75
	sub1.plot(xplot, [lgad_multiplication(x-th_passivation, 6.2, W1) for x in xplot], label='Standard', color='red')
	sub1.plot(xplot, [lgad_multiplication(x-th_passivation, 4.9, W9) for x in xplot], label='Shallow', color='black')
	
	sub1.axvspan(0, 75, facecolor='gold', )
	sub1.axvspan(75, 112+75, facecolor='lightsteelblue', )
#	sub1.axvspan(670+75, 2000, facecolor='bisque', alpha=0.7)
#	sub1.axvspan(299+75, 2000, facecolor='bisque', alpha=0.7)

	t=sub1.text(75/2, 10, 'Passivation', fontsize=10, color='black',ha="center", va="top", rotation='vertical') #, fontweight = 'bold')
	t1=sub1.text((112+75+75)/2, 10, '$n^+$-implant', fontsize=10, color='black',ha="center", va="top", rotation='vertical') #, 	#t.set_bbox(dict(facecolor='white', alpha=.9, linewidth=0))		
#	p3 = patches.FancyArrowPatch((670, 3), (2000, 3),linewidth=2, arrowstyle='<|-|>', mutation_scale=10, ec='gray')
#	sub1.add_patch(p3)

	p2 = patches.FancyArrowPatch((112+75, 8), (670+75, 8),linewidth=1, arrowstyle='<|-|>', mutation_scale=10, color='black')
	sub1.add_patch(p2)
	sub1.plot( [112+75,112+75],[2.3, 8], '--', color='gray')
	sub1.plot( [670+75,670+75],[6.4, 8], '--', color='gray')
	t=sub1.text(466, 8.6, 'Standard\ngain layer', fontsize=10, color='red',ha="center", va="center",) #, fontweight = 'bold')

	sub1.plot( [299+75,299+75],[5.1, 6], '--', color='gray')
	p1 = patches.FancyArrowPatch((112+75, 6), (299+75, 6),linewidth=1, arrowstyle='<|-|>', mutation_scale=10, color='black')
	sub1.add_patch(p1)
	t=sub1.text(200, 6.6, 'shallow\ngain layer', fontsize=10, color='black',ha="left", va="center",) #, fontweight = 'bold')

	fig2, sub2 = plt.subplots() 
	sub2.set(xlabel='depth (nm)')	
	fn='LGADs_absorption_500eV.txt'
	energy = int(re.search('_absorption_(.*)eV',fn).group(1))
	print('Photon Energy:', energy, 'eV')
	eventID, posZ, edep = get_simulated_data('data/'+fn)
	#print('Min hit position:',np.min(posZ), 'nm - Max:', np.max(posZ), 'nm')
	n,b,h = sub2.hist(posZ,500, label=str(energy)+'eV', histtype='step', )
	print(np.max(n))
	
	param, cov = curve_fit(exp, b[:-1],n, p0=[np.max(n),0.01])
	print(param)
	sub2.plot(xplot,exp(xplot, param[0],param[1]))

	fn='LGADs_absorption_1000eV.txt'
	energy = int(re.search('_absorption_(.*)eV',fn).group(1))
	print('Photon Energy:', energy, 'eV')
	eventID, posZ, edep = get_simulated_data('data/'+fn)
	#print('Min hit position:',np.min(posZ), 'nm - Max:', np.max(posZ), 'nm')
	n,b,h = sub2.hist(posZ,500, label=str(energy)+'eV', histtype='step', )
	print(np.max(n))
	
	param1, cov1 = curve_fit(exp, b[:-1],n, p0=[np.max(n),0.001])
	print(param1)
	sub2.plot(xplot,exp(xplot, param1[0],param1[1]))

	fig2.show()

	ax1 = sub1.twinx()
#	sub1[1].set_ylim([0.1,1])
	ax1.set_ylim([0,2])
	ax1.tick_params(direction="in")
	sub1.tick_params(direction="in")
	ax1.set(ylabel='Absorbed photons (AU)')
	ax1.plot(xplot,[exp(x, 1,param1[1]) if x>0 else 0 for x in xplot], ':', label='Absorption of\n1000 eV photons')
	ax1.plot(xplot,[exp(x, 1,param[1]) if x>0 else 0 for x in xplot], '--', label='Absorption of\n500 eV photons')
	ax1.legend(frameon=False, loc='center right', bbox_to_anchor=(0.95, 0.15))
	
	sub1.legend(frameon=False)
	fig1.show()
    
def compare_geant_lbl():  
    print('Compare the attenuation coefficient from LBL\nwith the Geant4 simulation')

    fig1, sub1 = plt.subplots() 
    sub1.set(xlabel='Energy (eV)',ylabel='Attenuation coefficient (1/nm)')	

    fig2, sub2 = plt.subplots() 
    sub2.set(xlabel='depth (nm)')	

    geantfiles = ['LGADs_absorption_200eV.txt',
    'LGADs_absorption_250eV.txt',
    'LGADs_absorption_300eV.txt',
    'LGADs_absorption_350eV.txt',
    'LGADs_absorption_400eV.txt',
    'LGADs_absorption_450eV.txt',
    'LGADs_absorption_500eV.txt',
    'LGADs_absorption_550eV.txt',
    'LGADs_absorption_600eV.txt',
    'LGADs_absorption_650eV.txt',
    'LGADs_absorption_700eV.txt',
    'LGADs_absorption_750eV.txt',
    'LGADs_absorption_800eV.txt',
    'LGADs_absorption_850eV.txt',
    'LGADs_absorption_900eV.txt',]
 
    xplot = np.arange(0,2000,10)
    coefficients = np.zeros(len(geantfiles))
    energies = np.zeros(len(geantfiles))

   
    for i,fn in enumerate(geantfiles):
	    energy = int(re.search('_absorption_(.*)eV',fn).group(1))
	    energies[i]= energy
	    print('Photon Energy:', energy, 'eV')
	    eventID, posZ, edep = get_simulated_data('data/'+fn)
	    #print('Min hit position:',np.min(posZ), 'nm - Max:', np.max(posZ), 'nm')
	    n,b,h = sub2.hist(posZ-75,500, label=str(energy)+'eV', histtype='step', )
	    
	    param, cov = curve_fit(exp, b[:-1],n, p0=[np.max(n),0.01])
	    print(param)
	    sub2.plot(xplot,exp(xplot, param[0],param[1]))
	    coefficients[i] = param[1]

    print(coefficients)
    fig2.show()

    nist = np.loadtxt('/mnt/sls_det_storage/eiger_data/lgad/Filippo_analysis/lgads_sim/lgads_montecarlo/data/lbl_att_coeff.txt')
    sub1.plot(energies,coefficients, 'o', label='Geant')
    sub1.plot(nist[:,0],1/(nist[:,1]*1000), label='LBL')
    sub1.set(xlim=(100,1000))

    sub1.legend()
    fig1.show()

def calculate_efficiency():
    
    #get the mu from lbl
    fig1, sub1 = plt.subplots() 
    sub1.set(xlabel='Energy (eV)',ylabel='Attenuation coefficient (1/nm)')	
    mu_lbl = np.loadtxt('data/lbl_att_coeff.txt')
    sub1.plot(mu_lbl[:,0],1/(mu_lbl[:,1]*1000), label='LBL')

    # mu from Geant4
    mu_geant = np.array([[200, 0.01680155],
                         [250, 0.01081149],
                         [300, 0.00718765],
                         [350, 0.00502069],
                         [400, 0.00372266], 
                         [450, 0.00279584], 
                         [500, 0.00215177], 
                         [550, 0.0017018],  
                         [600, 0.00136018], 
                         [650, 0.00110544], 
                         [700, 0.00092519], 
                         [750, 0.0007662],
                         [800, 0.00065365], 
                         [850, 0.00056055], 
                         [900, 0.00048302]])

    sub1.plot(mu_geant[:,0],mu_geant[:,1], label='Geant4')
    sub1.set(xlim=(100,1000))
    sub1.legend()
    fig1.show()

def linear(x,m,q):
	return x*m+q

def fit_calibration(x,y,p0,sigma):
	param, cov = curve_fit(linear, x, y, p0=p0,sigma=sigma)
	errs = np.sqrt(np.diag(cov))
	return np.concatenate((param,errs))	

def plot_best_charge_cloud():
    print('Plot the optimization of the charge cloud dimentions')
    W9_cloud = np.array([3.5,4,4.68,6,5.5,3,])
    W9_peak_900 = np.array([[880.45, 1.57],[878.31, 1.71],[872.17, 1.24],[864.05, 1.72],[868.88, 2.53],[882.18, 0.65]])
    W9_data_peak_900 = 875.55
    W9_peak_850 = np.array([[829.80,1.23],[827.96,0.62],[823.66,2.40],[814.56,1.71],[818.65,0.99],[832.70,0.82]])
    W9_data_peak_850 = 827.00
    W9_peak_800 = np.array([[780.27,0.61],[778.69,1.08],[773.64,1.73],[766.30,1.81],[767.29,2.07],[782.03,0.95]])
    W9_data_peak_800 = 771.26

    plot_off = 0.01

    fig1, sub1 = plt.subplots() 

    c_r = fit_calibration(W9_cloud,W9_peak_900[:,0]-W9_data_peak_900,[0,5],W9_peak_900[:,1])
    sub1.errorbar(W9_cloud, W9_peak_900[:,0]-W9_data_peak_900, yerr=W9_peak_900[:,1], color='black',
	marker = 'o',mfc='none', linestyle = 'none', capsize=3, capthick=1, label='Simulation - 900 eV')
    # sub1.plot([W9_cloud.min(), W9_cloud.max()], [W9_data_peak_900,W9_data_peak_900],label='Data - 900 eV',color='black')
    sub1.plot(np.sort(W9_cloud), [c_r[0]*e +c_r[1] for e in np.sort(W9_cloud)], '--', color='black')#, label=f'linear fit: y={c_r[0]:.2f}x {c_r[1]:.2f}')
    opt_900 = (-c_r[1])/c_r[0]

    c_r = fit_calibration(W9_cloud,W9_peak_850[:,0]-W9_data_peak_850,[0,5],W9_peak_850[:,1])
    sub1.errorbar(W9_cloud-plot_off, W9_peak_850[:,0]-W9_data_peak_850, yerr=W9_peak_850[:,1], color='blue',
	marker = 's',mfc='none', linestyle = 'none', capsize=3, capthick=1, label='Simulation - 850 eV')
    # sub1.plot([W9_cloud.min(), W9_cloud.max()], [W9_data_peak_850,W9_data_peak_850],label='Data - 850 eV',color='blue')
    sub1.plot(np.sort(W9_cloud-plot_off), [c_r[0]*e +c_r[1] for e in np.sort(W9_cloud-plot_off)], '--', color='blue')#, label=f'linear fit: y={c_r[0]:.2f}x {c_r[1]:.2f}')
    opt_850 = (-c_r[1])/c_r[0]

    c_r = fit_calibration(W9_cloud,W9_peak_800[:,0]-W9_data_peak_800,[0,5],W9_peak_800[:,1])
    sub1.errorbar(W9_cloud+plot_off, W9_peak_800[:,0]-W9_data_peak_800, yerr=W9_peak_800[:,1], color='green',
	marker = 'v',mfc='none', linestyle = 'none', capsize=3, capthick=1, label='Simulation - 800 eV')
    sub1.plot(np.sort(W9_cloud+plot_off), [c_r[0]*e +c_r[1] for e in np.sort(W9_cloud+plot_off)], '--', color='green')#, label=f'linear fit: y={c_r[0]:.2f}x {c_r[1]:.2f}')
    opt_800 = (-c_r[1])/c_r[0]
    opt_charge_cloud = np.average([opt_900,opt_850,opt_800])
    print('Optimal charge cloud:', opt_charge_cloud)

    ylim = sub1.get_ylim()
    sub1.plot([opt_charge_cloud,opt_charge_cloud], ylim,':')
    t=sub1.text(opt_charge_cloud+0.3,-14.6, 'Estimated value:\n'+f'{opt_charge_cloud:.2f}'+' $\mathrm{\mu m}$', fontsize=10, color='black', ha="center")

    sub1.plot([W9_cloud.min(), W9_cloud.max()],[0,0],color='red')
    sub1.tick_params(direction='in')
    sub1.set(xlabel='Charge cloud dimension ($\mathrm{\mu m}$)', ylabel='Simulated peak position - data (eV)',ylim=ylim)
    sub1.legend(frameon=False, title='Shallow LGAD', title_fontproperties={'weight':'bold'})
    fig1.show()
    
    W1_cloud = np.array([6,5.5,7,8,6.5,9,7.5,8.5,9.5,10])
    W1_peak_900 = np.array([[859.87,2.80],[863.30,1.36],[852.71,2.37],[845.26,1.64],[858.68,2.10],[837.81,4.33],[848.15,2.00],[841.30,4.25],[829.37,3.85],[825.69,5.03]])
    W1_data_peak_900 = 850.06
    W1_peak_850 = np.array([[810.91,2.91],[816.34,3.84],[803.24,3.78],[795.94,1.20],[807.73,4.62],[786.92,3.06],[799.31,4.46],[791.61,3.32],[779.00,3.24],[773.28,1.48]])
    W1_data_peak_850 = 795.38
    W1_peak_800 = np.array([[762.82,3.12],[765.19,1.75],[754.35,0.56],[748.78,3.33],[758.49,2.38],[736.31,3.14],[750.91,4.20],[742.46,3.57],[730.63,5.40],[726.28,3.23]])
    W1_data_peak_800 = 741.75

    fig2, sub2 = plt.subplots() 

    c_r = fit_calibration(W1_cloud,W1_peak_900[:,0]-W1_data_peak_900,[0,5],W1_peak_900[:,1])
    sub2.errorbar(W1_cloud, W1_peak_900[:,0]-W1_data_peak_900, yerr=W1_peak_900[:,1], color='black',
	marker = 'o',mfc='none', linestyle = 'none', capsize=3, capthick=1, label='Simulation - 900 eV')
    # sub1.plot([W1_cloud.min(), W1_cloud.max()], [W1_data_peak_900,W1_data_peak_900],label='Data - 900 eV',color='black')
    sub2.plot(np.sort(W1_cloud), [c_r[0]*e +c_r[1] for e in np.sort(W1_cloud)], '--', color='black')#, label=f'linear fit: y={c_r[0]:.2f}x {c_r[1]:.2f}')
    opt_900 = (-c_r[1])/c_r[0]

    c_r = fit_calibration(W1_cloud,W1_peak_850[:,0]-W1_data_peak_850,[0,5],W1_peak_850[:,1])
    sub2.errorbar(W1_cloud-plot_off, W1_peak_850[:,0]-W1_data_peak_850, yerr=W1_peak_850[:,1], color='blue',
	marker = 's',mfc='none', linestyle = 'none', capsize=3, capthick=1, label='Simulation - 850 eV')
    # sub1.plot([W1_cloud.min(), W1_cloud.max()], [W1_data_peak_850,W1_data_peak_850],label='Data - 850 eV',color='blue')
    sub2.plot(np.sort(W1_cloud-plot_off), [c_r[0]*e +c_r[1] for e in np.sort(W1_cloud-plot_off)], '--', color='blue')#, label=f'linear fit: y={c_r[0]:.2f}x {c_r[1]:.2f}')
    opt_850 = (-c_r[1])/c_r[0]

    c_r = fit_calibration(W1_cloud,W1_peak_800[:,0]-W1_data_peak_800,[0,5],W1_peak_800[:,1])
    sub2.errorbar(W1_cloud+plot_off, W1_peak_800[:,0]-W1_data_peak_800, yerr=W1_peak_800[:,1], color='green',
	marker = 'v',mfc='none', linestyle = 'none', capsize=3, capthick=1, label='Simulation - 800 eV')
    sub2.plot(np.sort(W1_cloud+plot_off), [c_r[0]*e +c_r[1] for e in np.sort(W1_cloud+plot_off)], '--', color='green')#, label=f'linear fit: y={c_r[0]:.2f}x {c_r[1]:.2f}')
    opt_800 = (-c_r[1])/c_r[0]
    opt_charge_cloud = np.average([opt_900,opt_850,opt_800])
    print('Optimal charge cloud:', opt_charge_cloud)

    ylim = sub2.get_ylim()
    sub2.plot([opt_charge_cloud,opt_charge_cloud], ylim,':')
    t=sub2.text(opt_charge_cloud+0.4,-30, 'Estimated value:\n'+f'{opt_charge_cloud:.2f}'+' $\mathrm{\mu m}$', fontsize=10, color='black', ha="center")

    sub2.plot([W1_cloud.min(), W1_cloud.max()],[0,0],color='red')
    sub2.tick_params(direction='in')
    sub2.set(xlabel='Charge cloud dimension ($\mathrm{\mu m}$)', ylabel='Simulated peak position - data (eV)',ylim=ylim)
    sub2.legend(frameon=False, title='Standard LGAD', title_fontproperties={'weight':'bold'})
    fig2.show()
    
def calculate_many_charge_collection():
    centre_value = 4.88
    sigma = 0.08

    start_value = round(centre_value - 6*sigma,1)
    stop_value = round(centre_value + 6*sigma,1)

    print(start_value,stop_value)
    steps = np.arange(start_value,stop_value+0.01,0.1)
    steps = [8.7,8.8,8.9]
    print(steps)
    for s in steps:
        print('step:',s)
        calculate_charge_collection(s)

def mult_efficiency():
    iterations = 500

    energies = np.arange(500,901,50)
    # energies = np.arange(500,551,100)
    efficiencies = np.zeros((iterations,len(energies)))

    model_g = GaussianModel()

    wafer = W1
    for i in range(iterations):
        print('Iteration:',i)

        # extract parameters from gaussian distributions
        Wc = copy.deepcopy(wafer)
        Wc['h_gain'] = np.random.normal(wafer['h_gain'], 1*wafer['h_gain_err'],1)[0]	
        Wc['depth_gl'] = np.random.normal(wafer['depth_gl'], 1*wafer['depth_gl_err'],1)[0]	
        Wc['charge_cloud'] = np.random.normal(wafer['charge_cloud'], 1*wafer['charge_cloud_err'],1)[0]	
            
        print('Wafer:',Wc)
        cl_dim_r = round(Wc['charge_cloud'] ,1)
        print(cl_dim_r, 'um')
        chcloud = np.loadtxt('data/charge_collection/charge_collection'+f'{cl_dim_r:.2f}'+'um_pp75um.txt').flatten()

        for i_ene,ene in enumerate(energies):

            abs_file = f'LGADs_absorption_{ene:.0f}eV.txt'
            # print(ene, 'eV')

            # get absorption depth
            eventID, posZ, edep = get_simulated_data('data/'+abs_file)

            # calculate multiplication
            th_passivation = np.min(posZ)
            mult_event = [lgad_multiplication_noiseless(x-th_passivation, e*1000, Wc) for x,e in zip(posZ,edep)]

            tot_events = len(posZ) * len(chcloud)

            pool = multiprocessing.Pool()
            g_e = np.array(list(pool.starmap(count_good_events, 
                                        zip(mult_event, repeat(chcloud), repeat(Wc)))))
            pool.close

            # print('tot:', tot_events, 'good', np.sum(g_e), np.sum(g_e)/tot_events*100,'%')
            efficiencies[i,i_ene] = np.sum(g_e)/tot_events

    fig1, sub1 = plt.subplots() 

    for i_ene,ene in enumerate(energies):
        n9,b9,h9 = sub1.hist(efficiencies[:,i_ene].flatten(), 200, label=str(ene)+'eV', histtype='step', )
        params_g = model_g.make_params(center=b9[np.argmax(n9)], amplitude=np.amax(n9), sigma=np.std(efficiencies[:,i_ene])) 
        result_g = model_g.fit(n9, params_g, x=(b9[1:] + b9[:-1])/2)
        sub1.plot((b9[1:] + b9[:-1])/2, result_g.best_fit, '-')
        print(result_g.params.valuesdict())
        print('Average:',np.average(efficiencies[:,i_ene]))
        print('std.dev',np.std(efficiencies[:,i_ene]))

    sub1.legend()
    fig1.show()

def count_good_events(ev, chcloud, W):
    return np.count_nonzero(np.random.normal(ev*chcloud, W['noise']*3.6) > W['noise']*5*3.6)