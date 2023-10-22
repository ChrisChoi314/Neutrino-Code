# Emir

This directory has all my work with the paper "Gravitational wave signal from massive gravity" https://arxiv.org/pdf/1208.5975.pdf by A. Emir Gumrukcuoglu, Sachiko Kuroyanagi, Chunshan Lin, Shinji Mukohyama, and Norihiro Tanahashi

All of the files (besides emir_func.py, *_copy.py files, *test.py, *_try.py), have a folder with the attachment "_figs" at the end of the name of the python file associated with them. These are the folders that contain the figures generated from the associated python file. 

# Files and what they do
- emir.py was my first look at the paper and has a lot of attempts at recreating the figures in the paper. Nothing is really well documented in that file so you can safely ignore my incoherent code in that file.
- emir_P.py includes my efforts to replicate the power spectrums shown in the paper, so this includes Fig. 4, 5, and 6 from the main paper. These correspond to the plots emir/emir_P_figs/fig2.pdf, emir/emir_P_figs/fig3.pdf, and emir/emir_P_figs/fig5.pdf respectively. The file emir/emir_P_figs/fig10.pdf has the upper bounds from the Claude de rham paper in 2017 https://arxiv.org/abs/1606.08462 and the oct 2023 paper https://arxiv.org/abs/2310.07469 plus the GR power spectrum plus the senstivity curve for the 15 year NANOGrav dataset.
- emir_S.py generates the transfer function mentioned in Eq. (47) of the original paper.
- emir_calc.py has my attempts to try to get the inverse Hubble function, and other stuff.
- emir_func.py has all of my functions and parameters for this directory.
- emir_func_copy.py is similiar to its sister file, except it has my failed attempts to try to plot the function that solves for scale factor given the wave vector k of the equation w_0^2 = H^2(a) (so basically a_k in Eq. (26)
- emir_h2omega.py has my various attempts to try to plot the energy density for the model in the paper. I pulled countless literature sources to try to relate the power spectrum with the energy density.
- emir_hasasia.py generates the sensitivty curve for the 15 year NANOGrav data. it works in conjunction with emir_hasasia_gen_dict.py
- emir_hasasia_gen_dict.py generates the sensitivities with a manually defined list of pulsars in the analysis.
- emir_kva.py generates Fig. 1 from the paper
- emir_mode.py has my analysis on the differential equation in Eq. (9) in the paper, and its solutions for different values of the mass and wave vector k.
- test.py has my random testing of various things.
- third_try.py and its copy are by failed attempts to find out why the inverse function for the hubble parameter weren't working. 
