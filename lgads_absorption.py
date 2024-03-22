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
import multiprocessing
from itertools import repeat
import multiprocessing

W1 = {'e_gain': 1,
      'h_gain': 0.253,
      'h_gain_err': 0.003,
      'th_impl': 112,  # nm
      'depth_gl': 670,  # nm
      'depth_gl_err': 12,  # nm
      'noise': 24 # e-  
      }

W9 = {'e_gain': 1,
      'h_gain': 0.377,
      'h_gain_err': 0.003,
      'th_impl': 107,  # nm
      'depth_gl': 299,  # nm 
      'depth_gl_err': 5,  # nm
      'noise': 25 # e-  
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
    'LGADs_absorption_200eV.txt',
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
    subW13.set(xlabel='Energy (eV)', ylabel='Counts', title='W13')       

    figW9comp, subW9comp = plt.subplots() 
    subW9comp.set(xlabel='Energy (eV)', ylabel='Counts', title='W9')
    
    comparison_files= ['data/W9_vrpre3500_E900eV.txt',
                    'data/W9_vrpre3500_E800eV.txt',
                    'data/W9_vrpre3500_E700eV.txt',
                    'data/W9_vrpre3700_E600eV.txt',
    ]
    plot_c = [50,45,45,40]    
    for jj,fn in enumerate(files):
        energy_plot[jj] = int(re.search('_absorption_(.*)eV',fn).group(1))
        print('Photon Energy:', energy_plot[jj], 'eV')
        eventID, posZ, edep = get_simulated_data('data/'+fn)
        #print('Min hit position:',np.min(posZ), 'nm - Max:', np.max(posZ), 'nm')
        if energy_plot[jj] %100 == 0:
            n,b,h = sub1.hist(posZ,500, label=str(energy_plot[jj])+'eV', histtype='step', )
        tot_photons = eventID[-1]+1
        print('Total photons:', tot_photons)
        recorded_photons = len(eventID)
        print('Recorded photons:', recorded_photons)
        QE_plot[jj] = recorded_photons/tot_photons
        print('QE:', QE_plot[jj])
        th_passivation = np.min(posZ)
        print('Passivation:', th_passivation)
        
        QE_plotW1[jj] = np.count_nonzero(posZ > W1['depth_gl']-th_passivation)/recorded_photons
        QE_plotW9[jj]= np.count_nonzero(posZ > W9['depth_gl']-th_passivation)/recorded_photons
        QE_plotW13[jj]= np.count_nonzero(posZ > W13['depth_gl']-th_passivation)/recorded_photons
        
        mult_event_W1 = [lgad_multiplication(x-th_passivation, e*1000, W1, 0*W1['noise']*3.6) for x,e in zip(posZ,edep)]
        mult_event_W9 = [lgad_multiplication(x-th_passivation, e*1000, W9, 0*W9['noise']*3.6) for x,e in zip(posZ,edep)]
        mult_event_W13 = [lgad_multiplication(x-th_passivation, e*1000, W13, 0*W13['noise']*3.6) 
                                                        for x,e in zip(posZ,edep)]

        foutw1 = open('data/mult_spectr_W1_E'+f'{energy_plot[jj]:0.0f}'+'eV.txt','w')
        for el in mult_event_W1: print(el, file=foutw1)
        foutw9 = open('data/mult_spectr_W9_E'+f'{energy_plot[jj]:0.0f}'+'eV.txt','w')
        for el in mult_event_W9: print(el, file=foutw9)
        foutw13 = open('data/mult_spectr_W13_E'+f'{energy_plot[jj]:0.0f}'+'eV.txt','w')
        for el in mult_event_W13: print(el, file=foutw13)
        
        if energy_plot[jj] %100 == 0:
            nm,bm,hm = subW9.hist(mult_event_W9,200, label=str(energy_plot[jj])+'eV', histtype='step', )
            nm,bm,hm = subW1.hist(mult_event_W1,200, label=str(energy_plot[jj])+'eV', histtype='step', )
            nm,bm,hm = subW13.hist(mult_event_W13,200, label=str(energy_plot[jj])+'eV', histtype='step', )

            if energy_plot[jj]  > 560:
                nm,bm,hm = subW9comp.hist(np.random.normal(mult_event_W9, W9['noise']*3.6),200, label=str(energy_plot[jj])+'eV', histtype='step', )

                for comp, ic in zip(comparison_files,plot_c):
                    if 'E'+str(energy_plot[jj]).replace('.0','')+'eV' in comp:
                        data = np.loadtxt(comp)
                        norm = np.max(nm)/np.max(data[:,1])
                        subW9comp.plot(data[ic:,0],data[ic:,1]*norm, '--', label=str(energy_plot[jj])+'eV data')

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
    sub2.plot(energy_plot,QE_plotW1, marker= '.',label='QE W1')    
    sub2.plot(energy_plot,QE_plotW9, marker= '.',label='QE W9')    
    sub2.plot(energy_plot,QE_plotW13, marker= '.',label='QE W13')    

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
    subW9comp.legend()
    figW9comp.show()
    
    print('Multiplied W1', QE_plotW1)
    print('Multiplied W9', QE_plotW9)
    print('Multiplied W13', QE_plotW13)

def charge_sharing_spectrum():
    energies = np.arange(200,901,50)
    wafers = ['W9']#['W1', 'W9', 'W13','W17']
    w_dics = [W9] #[W1,W9,W13,W17]
    tot_photons = 1e5 # number of simulated photons  
    QE=np.zeros((len(wafers),len(energies)))  
    QE_mult=np.zeros((len(wafers),len(energies)))  
        
    fig1, sub1 = plt.subplots() # weighting field
    
    sigma_cloud = 4.68 #um
    pp = 75 # um

    chcloud = np.loadtxt('/mnt/sls_det_storage/eiger_data/lgad/Filippo_analysis/lgads_sim/lgads_montecarlo/data/charge_collection4.68um_pp75um.txt')
    print(np.shape(chcloud))

    im = sub1.imshow(chcloud, extent=(0,pp,0,pp))
    sub1.plot([0,pp,pp,0,0],[0,0,pp,pp,0],'-', color='black')
    sub1.set(xlabel='x(um)',ylabel='y(um)', title='Charge collection')
    fig1.colorbar(im)
    fig1.show()
    
    comparison_files= ['data/W9_vrpre3500_E900eV.txt',
                'data/W9_vrpre3500_E800eV.txt',
                'data/W9_vrpre3500_E700eV.txt',
                'data/W9_vrpre3700_E600eV.txt',
    ]

    plot_c = [50,45,45,40]
    fig5, sub5 = plt.subplots()
    sub5.set(xlabel='Energy (eV)', ylabel='Counts', title='Comparison')           
    
    for iw,(w,ws) in enumerate(zip(wafers,w_dics)):
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
            
            QE_mult[iw,ie] = np.count_nonzero(data_noise > ws['noise']*5*3.6)/np.shape(data)[0]

            if e %100 == 0:    
                nm,bm,hm = sub2.hist(data_noise_nosh,200, label=str(e)+'eV', histtype='step')
                nm,bm,hm = sub3.hist(data_noise,200, label=str(e)+'eV', histtype='step')

            if e == 900:    
                nm,bm,hm = sub4.hist(data_noise_nosh,200, label=str(e)+'eV - NO charge sharing', histtype='step')
                nm,bm,hm = sub4.hist(data_noise,200, label=str(e)+'eV - with charge sharing', histtype='step')

            if w=='W9':
                for comp, ic in zip(comparison_files,plot_c):
                    if 'E'+str(e).replace('.0','')+'eV' in comp:
                        data = np.loadtxt(comp)
                        norm = np.max(nm)/np.max(data[:,1])
                        sub5.plot(data[ic:,0],data[ic:,1]*norm, '--', label=str(e)+'eV data')
                        sub5.hist(data_noise,200, label=str(e)+'eV sim', histtype='step')

        sub2.legend()    
        fig2.show()
        sub3.legend()    
        fig3.show()
        sub4.legend()    
        fig4.show()
        sub5.legend()    
        fig5.show()
        
    fig6, sub6 = plt.subplots()   
    sub6.set(xlabel='Energy (eV)', ylabel='Efficiency', title='Efficiency')            
    for iw,(w,ws) in enumerate(zip(wafers,w_dics)):
        sub6.plot(energies,QE[iw,:], marker= '.',label='QE - '+w)  
        sub6.plot(energies,QE_mult[iw,:], marker= '.',label='QE mult - '+w)  
        
    print(QE)        
    print(QE_mult)

    sub6.legend()    
    fig6.show()
       
def calculate_charge_collection():
    sigma_cloud = 4.68 #um
    pp = 75 # um
    Evgridx = np.arange(0,pp+.1,1)
    Evgridy = np.arange(0,pp+.1,1)    
    X, Y = np.meshgrid(Evgridx, Evgridy)
        
    integral_gauss2d_v = np.vectorize(integral_gauss2d)
    chcloud = integral_gauss2d_v(X,Y,sigma_cloud,1,pp)

    fig2, sub2 = plt.subplots() # weighting field    
    im = sub2.imshow(chcloud, extent=(0,pp,0,pp))
    sub2.plot([0,pp,pp,0,0],[0,0,pp,pp,0],'-', color='black')
    sub2.set(xlabel='x(um)',ylabel='y(um)', title='Charge collection')
    fig2.colorbar(im)
    fig2.show()

    fout = open('data/charge_collection'+f'{sigma_cloud:0.2f}'+'um_pp'+f'{pp:0.0f}'+'um'+'.txt','w')
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
	
	#en, measuredQE, signal>5sigma, column0*column1, photons absorbed after gain layer

	eff_W1 = np.array([ 
            [200,0.432448019,0.0000,0.0000,0.0002],
            [250,0.562035332,0.0001,0.0000,0.0032],
            [300,0.61335369,0.0010,0.0006,0.0224],
            [350,0.687525902,0.0072,0.0050,0.0686],
            [400,0.723115521,0.0328,0.0237,0.1419],
            [450,0.760093157,0.0956,0.0726,0.2297],
            [500,0.80063331,0.1948,0.1560,0.3221],
            [550,0.731947368,0.3095,0.2266,0.4054],
            [600,0.784586186,0.4240,0.3327,0.4854],
            [650,0.820386717,0.5254,0.4310,0.5573],
            [700,0.855498485,0.6069,0.5192,0.6120],
            [750,0.864894685,0.6764,0.5850,0.6652],
            [800,0.879920641,0.7326,0.6446,0.7070],
            [850,0.894954498,0.7811,0.6990,0.7432],
            [900,0.916747858,0.8216,0.7532,0.7740],
			])
	
	eff_W9 = np.array([
            [200,0.489865363,0.0000,0.0000,0.0701],
            [250,0.588656204,0.0005,0.0003,0.1795],
            [300,0.655403506,0.0051,0.0034,0.3188],
            [350,0.700047389,0.0249,0.0174,0.4468],
            [400,0.743308963,0.0763,0.0567,0.5541],
            [450,0.72288984,0.1787,0.1292,0.6404],
            [500,0.7747794,0.3128,0.2423,0.7112],
            [550,0.695852713,0.4485,0.311,0.7651],
            [600,0.746760343,0.5680,0.4241,0.8058],
            [650,0.779146386,0.6560,0.5111,0.8383],
            [700,0.807935326,0.7220,0.5833,0.8634],
            [750,0.821818386,0.7765,0.6382,0.8853],
            [800,0.848448683,0.8181,0.6941,0.9013],
            [850,0.847307341,0.8526,0.7224,0.9132],
            [900,0.858738425,0.8815,0.7570,0.9258],
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


	sub1[1].plot(QE[7:,0],QE[7:,1],':',color='gray', label='QE')
	sub1[1].plot(eff_W9[7:,0],eff_W9[7:,2],color='black', label='Shallow')
	sub1[1].plot(eff_W1[7:,0],eff_W1[7:,2],color='red', label='Standard')
	sub1[0].tick_params(bottom=True, top=False, left=True, right=True, direction="in")
	sub1[1].tick_params(bottom=True, top=False, left=True, right=True, direction="in", labelleft=False)
	
	ax1 = sub1[1].twinx()
	sub1[1].set_ylim([0.6,1])
	ax1.set_ylim([0.6,1])
	ax1.tick_params(direction="in")
	ax1.set(ylabel='Efficiency')

	sub1[0].legend(frameon=False)
	sub1[1].legend(frameon=False, loc='lower right')
#	sub1.set_xlim([450,1000])
	fig1.show()
	
def plot_noise_red():
	print('This is the gain reduction of LGADs')
	w13 = np.array([[2.3618, 0.6988,2.5504,0.2806],
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
    
    
    
    
    
    

