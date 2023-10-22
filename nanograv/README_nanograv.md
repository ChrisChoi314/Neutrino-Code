# Nanograv

This directory has all my work with the 15-year NANOGrav data set and my efforts towards a paper I am working on with  Murman Gurganidze, Tina Kahiashvili, and Jacob Magallanes

All of the files (besides *_func.py, and the copy file), have a folder with the attachment "_figs" at the end of the name of the python file associated with them. These are the folders that contain the figures generated from the associated python file. 

# Files and what they do
- blue_func.py has all of the functions and paramters related specifically to generating Fig 1 of the Blue tilted paper by Fujita, et al. https://arxiv.org/pdf/1808.02381.pdf 
- nanograv_func.py has all of the functions and parameters for everything else in this folder, except the ones related to generating Fig .1 in the blue tilted paper
- nanograv_masses.py has code that combine the Emir and Blue tilted paper in a single plot along with the 2-sigma posteriors of the NANOGrav data as a contour. The plot to look at is located in nanograv/nanograv_masses_figs/fig6.pdf
- ng_blue.py has my code that generates the 2-sigma contor of NANOGrav data, and just the energy densities of the model from Blue tilted paper.
- ng_blue (copy).py has my code that has everything from the ng_blue.py paper except it has a lot more extranneous testing code that I didn't want to get rid of.
- ng_emir.py has my code that generates the 2-sigma contor of NANOGrav data, and just the energy densities of the model from the Emir paper https://arxiv.org/pdf/1208.5975.pdf
- test.py has some of my random tests to verify stuff
