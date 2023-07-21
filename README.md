# Neutrino-Code
This is code I write for my research with Dr. Kahniashvili regarding the effect free streaming neutrinos in the early universe radiation-dominated era have on the gravitational waves generated from the QCD Electroweak era.

Also includes the research done on massive gravity, if gravitons were actually massive and not massless, and its affects on the power spectrum and whether it would be observable at a certain range of frequencies. 

DIRECTORY Layout
----------------------------
```
.
├── blue                            # Files and folders related to Mukohyama's paper 'Blue-tilted primordial gravitational waves from massive gravity', https://arxiv.org/pdf/1808.02381.pdf 
│   ├── Massive-Calculation.nb      # Mathematica notebook for calculations related to the blue tilted paper
│   ├── massive_copy.py             # Initial version of massive.py, filled with experimental code
│   ├── massive_figs                # Figures generated by massive.py
│   │   └── ...
│   ├── massive.py                  # Produces a numerical solution for the gravitational wave evolution in the paper for all 3 regimes
│   ├── power_figs
│   │   └── ...
│   └── power.py
├── emir                            # Files and folders related Emir Gumrukcuoglu's paper 'Gravitational wave signal from massive gravity', https://arxiv.org/pdf/1208.5975.pdf  
│   ├── emir_calc_figs
│   │   └── ...
│   ├── emir_calc.py
│   ├── emir_func_copy.py
│   ├── emir_func.py
│   ├── emir_hasasia
│   │   └── ...
│   ├── emir_hasasia_gen_dict.py
│   ├── emir_hasasia.py
│   ├── emir_kva_figs
│   │   └── ...
│   ├── emir_kva.py
│   ├── emir_mode_figs
│   │   └── ...
│   ├── emir_mode.py
│   ├── emir_P_figs
│   │   └── ...
│   ├── emir_P.py
│   ├── emir.py
│   ├── emir_S_figs
│   │   └── ...
│   ├── emir_S.py
│   ├── __pycache__
│   │   └── ...
│   ├── test.py
│   ├── third_try (copy).py
│   └── third_try.py
├── README.md
└── weinberg                        # Files and folders related to Weinberg's paper 'Damping of Tensor Modes in Cosmology', https://arxiv.org/pdf/astro-ph/0306304.pdf 
    ├── analytical_figs
    │   └── ...
    ├── analytical.py
    ├── general_wavelength_figs
    │   └── ...
    ├── general_wavelength.py
    ├── short_wavelength_figs
    │   └── ...
    └── short_wavelength.py
```

How to run the emir/emir_hasasia.py program:

You can just run emir/emir_P.py to generate the plot; I saved an .npz file that contains the NANOGrav frequencies and sensitivity in np arrays. But if you want to generate the sensitivity curve data yourself, you will need to run both emir/emir_hasasia.py and emir/emir_hasasia_gen_dict.py, perhaps multiple times. 

Firstly, note that some packages necessary for running emir/emir_hasasia.py (specifically enterprise-pulsar) can't be easily downloaded on android or the M1/M2 architecture for Macbooks using pip or conda. 

Otherwise, you may proceed with a Windows, Linux, or intel era Apple device. 

Download the 15-year NANOGrav data from the website https://zenodo.org/record/7967585 and have the directory structure as follows:
```
.
├── ...
├── Neutrino-Code           # This repository
│   └──  ...                 
└── NANOGrav_15yr_v1.0.1    # The extracted folder for the NANOGrav data

```
You may download it anywhere else, but make sure to change lines 23, 24, 25 in emir/emir_hasasia.py accordingly. 

Unless you have a computer that has over 30 GB, you're going to want to download the data in chunks. 

First, run ../../emir -- TBC