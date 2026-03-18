from latex_to_python import latex_to_python
import numpy as np

#Expression for TaylorT3 phase taken from Eq. (3.10a) of 0907.0700
phi_TaylorT3_latex = r'phi_ref -\frac{1}{\nu \theta^5}\left[1+ \left( \frac{3715}{8064}+\frac{55}{96}\nu \right)\theta^2 -\frac{3\pi}{4}\theta^3 + \left(\frac{9275495}{14450688}+\frac{284875}{258048 }\nu + \frac{1855}{2048 }\nu^2\right)\theta^4 \right.\nonumber\\ &+& \left (\frac {38645}{21504} - \frac{65}{256 }\nu \right ) \ln \left ( \frac {\theta}{\theta_{\rm lso}} \right ) \pi\theta^5 + \left\{ \frac {831032450749357}{57682522275840} - \frac {53}{40}\pi^2 + \left (- \frac {126510089885}{4161798144} + \frac {2255}{2048} \pi^2 \right ) \nu \right. \nonumber \\ &-& \frac {107}{56} \gamma + \left.\left. \frac {154565}{1835008} \nu^2 - \frac {1179625}{1769472} \nu^3 - \frac {107}{56}\log(2\theta) \right \} \theta^6 + \left ( \frac {188516689}{173408256} + \frac {488825}{516096} \nu - \frac {141769}{516096} \nu^2 \right )\pi\theta^7 \right]'

#Create a dictionary where the keys are the variables as written in Latex
#and the entries are the python names of those variables
variables_TaylorT3 = {r'\theta_{\rm lso}': 'theta_lso',
                      r'\theta': 'theta',
                      r'\nu': 'nu',
                      r'\pi': 'np.pi',
                      r'\gamma': 'np.euler_gamma'}

#extra stuff to change
replace_extra_TaylorT3 = {r'\log': r'np.log',
                          r'\ln' : r'np.log'}

print('phi =', latex_to_python(phi_TaylorT3_latex, variables_TaylorT3, replace_extra=replace_extra_TaylorT3), '\n\n')

#fixed formated strings
phi_TaylorT3_python = 'phi_ref-(1/(nu*theta**5))*(1+((3715/8064)+(55/96)*nu)*theta**2-((3*np.pi)/4)*theta**3+((9275495/14450688)+(284875/258048)*nu+(1855/2048)*nu**2)*theta**4+((38645/21504)-(65/256)*nu)*np.log((theta/theta_lso))*np.pi*theta**5+((831032450749357/57682522275840)-(53/40)*np.pi**2+(-(126510089885/4161798144)+(2255/2048)*np.pi**2)*nu-(107/56)*np.euler_gamma+(154565/1835008)*nu**2-(1179625/1769472)*nu**3-(107/56)*np.log(2*theta))*theta**6+((188516689/173408256)+(488825/516096)*nu-(141769/516096)*nu**2)*np.pi*theta**7)'
