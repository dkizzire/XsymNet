Reason there are three stages in write_TOPAS_experimental_pawley

Stage 1: 
    1. has continue_after_convergence turned on and goes for many iterations
    2. Ability to refine the lattice parameters by large amounts means that the TOF values in load_hkl_m_2th I 
        can be outside of the start and finish ranges you initially stages
        -> this provides the need for stage 2

Stage 2:
    1. We copy the output of stage 1 and turn it into the input for stage 2 with the following changes.
    2. We turn off continue_after_convergence and set the iterations to 500
    3. We delete the existing load_hkl_m_2th since the TOF values are outside of our intended range and let TOPAS regenerate it for us 
    
    -> After stage 2, we have an excellent Rwp with TOF values in our intended range, but the scale is not good for comparing against our ML data so we need stage 3.

Stage 3:
    1. We copy the output of stage 2 and turn it into stage 3 with the following changes.
    2. We set iterations to 0
    3. We set the scale line to : scale @ 1.0
    4. We output the simulated xy and hkl I files

    ->  now, the output of stage 3 will have a "bad" Rwp since we messed with the scale, but the data will be perfect for 
    using in our ML algorithm. 

For the next step of generating the ML data, there are a few things we need to take from Stage 3 output to make our ML data generation Topas input files

We need to copy the following lines:
1.  TOF_x_axis_calibration
2.  The background line : bkg
3.  tof_sample_peakshape

Putting these three lines in all of our topas input files for ML data generation will ensure that we have consistent simulated data since the ML data generation input files (randomstates) all have 0 iterations and none of the three lines above will change during their running. 