from enterprise.pulsar import Pulsar as ePulsar
import json
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
# matplotlib inline

import glob
import pickle
import json
import sys

import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.skymap as hsky

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = [5, 3]
# mpl.rcParams['text.usetex'] = True


pardir = '../NANOGrav_15yr_v1.0.1/narrowband/par/'
timdir = '../NANOGrav_15yr_v1.0.1/narrowband/tim/'
noise_dir = '../NANOGrav_15yr_v1.0.1/narrowband/noise/'
pars = sorted(glob.glob(pardir+'*.par'))
tims = sorted(glob.glob(timdir+'*.tim'))
noise_files = sorted(glob.glob(noise_dir+'*wb.pars.txt'))

psr_list = ['B1855+09', 'B1937+21', 'B1937+21ao', 'B1937+21gbt', 'B1953+29', 'J0023+0923', 'J0030+0451', 'J0340+4130', 'J0406+3039', 'J0437-4715', 'J0509+0856', 'J0557+1551', 'J0605+3757', 'J0610-2100', 'J0613-0200', 'J0614-3329', 'J0636+5128', 'J0645+5158', 'J0709+0458', 'J0740+6620', 'J0931-1902', 'J1012+5307', 'J1012-4235', 'J1022+1001', 'J1024-0719', 'J1125+7819', 'J1312+0051', 'J1453+1902', 'J1455-3330', 'J1600-3053', 'J1600-3053gbt', 'J1614-2230', 'J1630+3734', 'J1640+2224', 'J1643-1224', 'J1643-1224gbt', 'J1705-1903', 'J1713+0747',
            'J1713+0747ao', 'J1713+0747gbt', 'J1719-1438', 'J1730-2304', 'J1738+0333', 'J1741+1351', 'J1744-1134', 'J1745+1017', 'J1747-4036', 'J1751-2857', 'J1802-2124', 'J1811-2405', 'J1832-0836', 'J1843-1113', 'J1853+1303', 'J1903+0327', 'J1903+0327ao', 'J1909-3744', 'J1909-3744gbt', 'J1910+1256', 'J1911+1347', 'J1918-0642', 'J1923+2515', 'J1944+0907', 'J1946+3417', 'J2010-1323', 'J2017+0603', 'J2033+1734', 'J2043+1711', 'J2124-3358', 'J2145-0750', 'J2214+3000', 'J2229+2643', 'J2234+0611', 'J2234+0944', 'J2302+4442', 'J2317+1439', 'J2322+2057']


def get_psrname(file, name_sep='_'):
    return file.split('/')[-1].split(name_sep)[0]


# print(noise_files)
pars = [f for f in pars if get_psrname(f) in psr_list]
tims = [f for f in tims if get_psrname(f) in psr_list]
noise_files = [f for f in noise_files if get_psrname(f, '.') in psr_list]

with open('emir/emir_hasasia/noise_narrowband.json', 'r') as f:
    noise = json.load(f)
print('cp 0')

'''

ePsrs = []
print(len(pars), len(tims))
i = 0

for par, tim in zip(pars, tims):
    if i in range(60, 80):
        ePsr = ePulsar(par, tim,  ephem='DE436')
        ePsrs.append(ePsr)
        print('\rPSR {0} complete, idx = {1}'.format(ePsr.name, i), end='', flush=True)
    i+=1



np.savez("emir/emir_hasasia/ePulsars60_75.npz", data60_75=ePsrs)
print(ePsrs)
sys.exit()

'''

outfile1 = np.load('emir/emir_hasasia/ePulsars0_19.npz',allow_pickle=True)
outfile2 = np.load('emir/emir_hasasia/ePulsars20_39.npz',allow_pickle=True)
outfile3 = np.load('emir/emir_hasasia/ePulsars40_59.npz',allow_pickle=True)
outfile4 = np.load('emir/emir_hasasia/ePulsars60_75.npz',allow_pickle=True)


ePsrs = list(np.concatenate([outfile1['data0_19'], outfile2['data20_39'], outfile3['data40_59'], outfile4['data60_75']]))
# ePsrs = list(outfile4['data60_75'])

del outfile1
del outfile2
del outfile3
del outfile4


def make_corr(psr):
    N = psr.toaerrs.size
    corr = np.zeros((N, N))
    _, _, fl, _, bi = hsen.quantize_fast(psr.toas, psr.toaerrs,
                                         flags=psr.flags['f'], dt=1)
    keys = [ky for ky in noise.keys() if psr.name in ky]
    backends = np.unique(psr.flags['f'])
    sigma_sqr = np.zeros(N)
    ecorrs = np.zeros_like(fl, dtype=float)
    for be in backends:
        mask = np.where(psr.flags['f'] == be)
        key_ef = '{0}_{1}_{2}'.format(psr.name, be, 'efac')
        key_eq = '{0}_{1}_log10_{2}'.format(psr.name, be, 'equad')
        sigma_sqr[mask] = (noise[key_ef]**2 * (psr.toaerrs[mask]**2)
                           + (10**noise[key_eq])**2)
        mask_ec = np.where(fl == be)
        key_ec = '{0}_{1}_log10_{2}'.format(psr.name, be, 'ecorr')
        ecorrs[mask_ec] = np.ones_like(mask_ec) * (10**noise[key_ec])
    j = [ecorrs[ii]**2*np.ones((len(bucket), len(bucket)))
         for ii, bucket in enumerate(bi)]

    J = sl.block_diag(*j)
    corr = np.diag(sigma_sqr) + J
    return corr


rn_psrs = {'B1855+09': [10**-13.7707, 3.6081],
           'B1937+21': [10**-13.2393, 2.46521],
           'J0030+0451': [10**-14.0649, 4.15366],
           'J0613-0200': [10**-13.1403, 1.24571],
           'J1012+5307': [10**-12.6833, 0.975424],
           'J1643-1224': [10**-12.245, 1.32361],
           'J1713+0747': [10**-14.3746, 3.06793],
           'J1747-4036': [10**-12.2165, 1.40842],
           'J1903+0327': [10**-12.2461, 2.16108],
           'J1909-3744': [10**-13.9429, 2.38219],
           'J2145-0750': [10**-12.6893, 1.32307],
           }

Tspan = hsen.get_Tspan(ePsrs)

fyr = 1/(365.25*24*3600)
freqs = np.logspace(np.log10(1/(5*Tspan)), np.log10(2e-7), 600)

# psrs = []
thin = 10
i = 60

#for ePsr in ePsrs:
#    print(ePsr.toaerrs.size, f'this is index {i}')
#    i +=1
#sys.exit(0)

'''
for ePsr in ePsrs:
    #if i < 56:
    #    i+=1
    #    continue
    corr = make_corr(ePsr)[::thin, ::thin]
    plaw = hsen.red_noise_powerlaw(A=9e-16, gamma=13/3., freqs=freqs)
    if ePsr.name in rn_psrs.keys():
        Amp, gam = rn_psrs[ePsr.name]
        plaw += hsen.red_noise_powerlaw(A=Amp, gamma=gam, freqs=freqs)

    corr += hsen.corr_from_psd(freqs=freqs, psd=plaw,
                            toas=ePsr.toas[::thin])
    psr = hsen.Pulsar(toas=ePsr.toas[::thin],
                    toaerrs=ePsr.toaerrs[::thin],
                    phi=ePsr.phi, theta=ePsr.theta,
                    N=corr, designmatrix=ePsr.Mmat[::thin, :])
    psr.name = ePsr.name
    np.savez(f'emir/emir_hasasia/psrs_files/psr{i}.npz', data=[psr])
    del ePsr
    #print('\rPSR {0} complete, idx {1}'.format(psr.name, i), end='', flush=True)
    print('PSR {0} complete, idx {1}'.format(psr.name, i))

    i+=1

sys.exit(0)
'''
specs = []
for i in range(0,76):
# for p in psrs:
    if i == 37:
        i += 1
        continue
    outfile = np.load(f'emir/emir_hasasia/psrs_files/psr{i}.npz',allow_pickle=True)
    p = outfile['data'][0]
    del outfile
    sp = hsen.Spectrum(p, freqs=freqs)
    _ = sp.NcalInv
    specs.append(sp)
    # print('\rPSR {0} complete'.format(p.name), end='', flush=True)
    print('PSR {0} complete: idx {1}'.format(p.name, i))
ng11yr_sc = hsen.GWBSensitivityCurve(specs)

np.savez("emir/emir_hasasia/nanograv_sens_full.npz", freqs = ng11yr_sc.freqs, sens = ng11yr_sc.h_c)

plt.loglog(ng11yr_sc.freqs, ng11yr_sc.h_c)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Characteristic Strain, $h_c$')
plt.title('NANOGrav 15-year Data Set Sensitivity Curve')
plt.grid(which='both')
# plt.ylim(1e-15,9e-12)
# plt.savefig('emir/emir_hasasia/fig2.pdf')
plt.show()
