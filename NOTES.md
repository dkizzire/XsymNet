# TOPAS NOTES

-  Make sure to use dayton configured hkl outputs from topas.inc to gain double point precision in intensity output
-  When creating the perturbed hkl lists for the ML step, take the background from the P1 primitive pawley fit and use that background in topas for all perturbed structures
    1.  Think about using the per structure background - useful if using the calculated xy instead of hkl list for ML

## Jacs 2016 Rietveld Refinement steps
1. The best Rietveld fit was then determined by a process of repeated local minimization (RLM) from random starting values.
2. Experimental peak shapes, instrument calibration constants, and the background description were fixed during this process. 
3. For each candidate model, 5000 rapid least-squares iterations were initially performed with mode amplitudes fixed at zero and only symmetry-constrained cell parameters refining.
4. After each convergence, the variable cell parameters were randomized by ±1% and refined back to convergence. 
5. In a second cycle, a further 2000 least-squares iterations were performed with cell parameters randomized by ±0.1% around these predetermined values and all mode amplitudes allowed to vary.
6. To help convergence, we applied a weak restraint to keep nonimportant mode amplitudes close to zero.
7. After each convergence, mode amplitudes were reset to random values between −0.1 and +0.1.
8. Several repeat runs were performed and Rwp differences of <0.1% were typically found for a given candidate model between different runs 
9. For final Rietveld refinements used to extract detailed structural information, X-ray and neutron data were simultaneously fitted to dmin = 0.7 Å. 
10. Effective polyhedral tilt angles in each final structure were determined using GASP.

# Experimental Pawley Refinement
1. After a long refinement with continue_after_convergence and vc macro, the hkl list will have d-spacings outside of the initial start_X finish_X bounds
   1. To fix this, take the topas output with optimized parameters, delete the hkl list from hkl_Is section, and run another single refinement with continue_after_convergence turned off. This will regenerate the hkl list with d_spacings inside the start_X finish_X bounds. 


# BEQ  NOTES
1. Beq has greater effect at low TOF (low d (high Q)) and a lesser effect at high TOF (its an exponential decay)
2. Center of unit cell will have a smaller beq than the corners (or the pivot points) 
   1. Oxygens will have a larger beq range than the metals 
   2. If something is edge sharing, it will have lower beq than something that is corner sharing 