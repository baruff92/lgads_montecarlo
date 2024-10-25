import numpy as np
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
	  'charge_cloud': 7.36, # um
	  'charge_cloud_err': 0.15  
	  }

W9 = {'e_gain': 1,
	  'h_gain': 0.377,
	  'h_gain_err': 0.003,
	  'th_impl': 107,  # nm
	  'depth_gl': 299,  # nm 
	  'depth_gl_err': 5,  # nm
	  'noise': 22.3, # e-  
	  'charge_cloud': 4.88, # um
	  'charge_cloud_err': 0.08  
	  }

def get_simulated_data(txt_fname):
	data = np.loadtxt(txt_fname)
	return data[:,0], data[:,1], data[:,2]

def lgad_multiplication_noiseless(x, e, w):
	if x < 0: return 0
	if x >= 0 and x<= w['th_impl']: return e*w['h_gain']
	if x > w['th_impl'] and x < w['depth_gl']: 
		m = e*(w['e_gain']-w['h_gain'])/(w['depth_gl']-w['th_impl'])
		q = e*w['h_gain']-m*w['th_impl']
		return x*m + q
	if x>= w['depth_gl']: return e*w['e_gain']

def mult_efficiency():
	iterations = 1

	energies = np.arange(500,901,50)
	efficiencies = np.zeros((iterations,len(energies)))

	model_g = GaussianModel()

	wafer = W1
	for i in range(iterations):
		print('Iteration:',i)

		# extract parameters from gaussian distributions
		Wc = copy.deepcopy(wafer)
		Wc['h_gain'] = np.random.normal(wafer['h_gain'], wafer['h_gain_err'],1)[0]	
		Wc['depth_gl'] = np.random.normal(wafer['depth_gl'], wafer['depth_gl_err'],1)[0]	
		Wc['charge_cloud'] = np.random.normal(wafer['charge_cloud'], wafer['charge_cloud_err'],1)[0]	
			
		print('Wafer:',Wc)
		cl_dim_r = round(Wc['charge_cloud'] ,1)
		print(cl_dim_r, 'um')

		# chcloud is a matrix that includes the charge collection efficiency over the surface of the pixel
		chcloud = np.loadtxt('data/charge_collection/charge_collection'+f'{cl_dim_r:.2f}'+'um_pp75um.txt').flatten()

		for i_ene,ene in enumerate(energies):

			# this file contains a list of simulated photon events
			# listing their absorption depth and deposited energy
			abs_file = f'LGADs_absorption_{ene:.0f}eV.txt'
			eventID, posZ, edep = get_simulated_data('data/'+abs_file)

			# calculate multiplication
			th_passivation = np.min(posZ)
			mult_event = [lgad_multiplication_noiseless(z-th_passivation, e*1000, Wc) for z,e in zip(posZ,edep)]

			pool = multiprocessing.Pool()
			g_e = np.array(list(pool.starmap(count_good_events, 
										zip(mult_event, repeat(chcloud), repeat(Wc)))))
			pool.close

			tot_events = len(posZ) * len(chcloud)
			efficiencies[i,i_ene] = np.sum(g_e)/tot_events

	fig1, sub1 = plt.subplots() 

	for i_ene,ene in enumerate(energies):
		n,b,h = sub1.hist(efficiencies[:,i_ene].flatten(), 200, label=str(ene)+'eV', histtype='step', )
		params_g = model_g.make_params(center=b[np.argmax(n)], amplitude=np.amax(n), sigma=np.std(efficiencies[:,i_ene])) 
		result_g = model_g.fit(n, params_g, x=(b[1:] + b[:-1])/2)
		sub1.plot((b[1:] + b[:-1])/2, result_g.best_fit, '-')
		print(result_g.params.valuesdict())
		print('Average:',np.average(efficiencies[:,i_ene]))
		print('std.dev',np.std(efficiencies[:,i_ene]))

	sub1.legend()
	fig1.show()

def count_good_events(ev, chcloud, W):
	# calculate the fraction of collected charge (ev) 
	# depending on the incident position, over the entire pixel surface (= ev*chcloud)
	# then add a random gaussain blurring to each of these cases
	# finally count how many of these are > 5sigma
	return np.count_nonzero(np.random.normal(ev*chcloud, W['noise']*3.6) > W['noise']*5*3.6)

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

	# the arrays contain:
	# energy(eV)    mult_eff+-sigma  mult_eff*QE+-sigma    
	eff_W1 = np.array([ 
	[500,   0.185617423145022,	0.00395393604638716,	0.146212023810058,	0.0035],
	[550,   0.292433165629935,	0.00480834716706391,	0.208768048786754,	0.0041],
	[600,   0.401356891065869,	0.00502382587679762,	0.307308241011975,	0.0052],
	[650,   0.499794038512569,	0.00475720072797992,	0.399718554641455,	0.0060],
	[700,   0.581966760470757,	0.00440440856561319,	0.484031593122597,	0.0065],
	[750,   0.656214520436835,	0.00389643263829487,	0.553422804500403,	0.0069],
	[800,   0.715371943847391,	0.0033735567025035,	0.618213461498041,	0.0073],
	[850,   0.765702404405493,	0.00287009447243025,	0.667027039613118,	0.0081],
	[900,   0.808457738129607,	0.00246838815631012,	0.717702812217162,	0.0085],
			])
	
	eff_W9 = np.array([
	[500,   0.41707,	0.00340,	0.328530152668361,	0.0046],
	[550,   0.54159,	0.00328,	0.386639764467248,	0.0049],
	[600,   0.63879,	0.00281,	0.489105619928374,	0.0060],
	[650,   0.71224,	0.00233,	0.569624894870816,	0.0069],
	[700,   0.77006,	0.00188,	0.640474859100529,	0.0073],
	[750,   0.81843,	0.00150,	0.690226479859595,	0.0077],
	[800,   0.85532,	0.00123,	0.739152641056032,	0.0081],
	[850,   0.88459,	0.00102,	0.770592893405632,	0.0090],
	[900,   0.91031,	0.00087,	0.808124637147501,	0.0093],
			])

	QE = np.array([
	[200,   0.461156691,	0.00977576136679631],
	[250,   0.575345768,	0.00839697231016186],
	[300,   0.634378598,	0.00729056977864042],
	[350,   0.6937866455,	0.0251521818996094],
	[400,   0.733212242,	0.0109806848523539],
	[450,   0.7414914985,	0.00801575980832424],
	[500,   0.787706355,	0.0090973789458125],
	[550,   0.7139000405,	0.00788334918510645],
	[600,   0.7656732645,	0.00881979534523405],
	[650,   0.7997665515,	0.00929281539599846],
	[700,   0.8317169055,	0.0092029212667156],
	[750,   0.8433565355,	0.0092345885920432],
	[800,   0.864184662,	0.00933388716673632],
	[850,   0.8711309195,	0.0100714154097682],
	[900,   0.8877431415,	0.0101864783559455]]    )

	sub1[1].errorbar(QE[7:,0],QE[7:,1]*100,yerr=QE[7:,2]*100,marker = '.',mfc='none', color='grey', linestyle = ':',capsize=3, capthick=1, label='Quantum Efficiency (QE)')
	sub1[1].errorbar(eff_W9[1:,0],eff_W9[1:,3]*100,yerr=eff_W9[1:,4]*100, marker = 'none',mfc='none', color='black', linestyle = '--',capsize=3, capthick=1, label='Total efficiency - Shallow')
	sub1[1].errorbar(eff_W1[1:,0],eff_W1[1:,3]*100,yerr=eff_W1[1:,4]*100,marker = 'none',mfc='none', color='red', linestyle = '--',capsize=3, capthick=1, label='Total efficiency - Standard')
	sub1[0].tick_params(bottom=True, top=True, left=True, right=True, direction="in")
	sub1[1].tick_params(bottom=True, top=False, left=True, right=True, direction="in", labelleft=False)
	
	ax1 = sub1[1].twinx()
	sub1[1].set_ylim([0.1*100,100])
	ax1.set_ylim([0.1*100,100])
	ax1.tick_params(direction="in")
	ax1.set(ylabel='Efficiency (%)')

	sub1[0].legend(frameon=False)
	sub1[1].legend(frameon=False, loc='lower right')
	fig1.show()
