# Blue

This directory has all my work with the paper "Blue-tilted Primordial Gravitational Waves from Massive Gravity" https://arxiv.org/pdf/1808.02381.pdf by Tomohiro Fujita, Sachiko Kuroyanagi, Shuntaro Mizuno, and Shinji Mukohyama.

All of the files, besides blue_func.py or the copies, have a folder with the attachment "_figs" at the end of the name associated with them. These are the folders that contain the figures generated from the associated python file. 

# Files and what they do
- Massive-Calculation.nb is a mathematica file that solves for the Bunch-Davies vacuum condition, detailed in eq. (6) of the paper.
- blue_emir.py plots Fig. 1 of the paper, which shows the energy density of the gravitational waves with certain mass and tau_m/tau_r values specified in the paper. Much of this is taken from the paper https://arxiv.org/pdf/1407.4785.pdf. This corresponds to the file blue/blue_emir_figs/fig4.pdf
- blue_emir (copy).py has some extra exploration on this plot that I wanted to save but didn't want in the main file to not clutter it up
- blue_func.py contains the functions and parameter values that the blue_emir.py file uses. Much of these are taken from that paper.
-  massive.py was my first analysis of the paper, and generates many plots related to the behavior of the mode function that is the solution to the differential equation of Eq. (3)
-  massie_copy.py probably has a lot of other stuff related to the massive.py file that I wanted to save for later.
-  power.py looks into the dimensionless power spectrum of the gravitational waves in the paper, mostly the numerical one calculated directly from the mode function as given in Eq. (7) of the main paper.
-  test.py has a bunch of my random tests along the way, not anything that I wanted to document in a serious analysis, but something I wanted to check the shape or behavior of, or maybe calculating some scale factor at some conformal time, something like that.
