' This file is included from TOPAS.INC
' Place local standard macros here


'Takes non-refinable undistorted cell parameters, and variable strains as input.
'Returns strained cell parameters
macro straincell(pa0, pb0, pc0, pal0, pbe0, pga0)
{
	prm  !radian  57.2957795
	prm  palpha0 = pal0/radian;
	prm  pbeta0  = pbe0/radian;
	prm  pgamma0 = pga0/radian;
	'parent-lattice basis vectors in cartesian coords
	prm  zzz  = (Cos(palpha0)- Cos(pbeta0)*Cos(pgamma0))/Sin(pgamma0);
	prm  a0x = pa0;
	prm  b0x = pb0*Cos(pgamma0);
	prm  b0y = pb0*Sin(pgamma0);
	prm  c0x = pc0*Cos(pbeta0);
	prm  c0y = pc0*zzz;
	prm  c0z = pc0*Sqrt(1 - Cos(pbeta0)^2 - zzz^2);
}

'Use basis-transformation matrix to transform from parent-cell to supercell parameters
macro celltransform(t11, t12, t13, t21, t22, t23, t31, t32, t33)
{
	prm  !radian  57.2957795
	'parent-lattice basis vectors in cartesian coords
	prm  pv11 = a0x*(pe1 + 1);
	prm  pv21 = a0x*(pe6/2);
	prm  pv31 = a0x*(pe5/2);
	prm  pv12 = b0x*(pe1 + 1) + b0y*(pe6/2);
	prm  pv22 = b0x*(pe6/2)   + b0y*(pe2 + 1);
	prm  pv32 = b0x*(pe5/2)   + b0y*(pe4/2);
	prm  pv13 = c0x*(pe1 + 1) + c0y*(pe6/2) + c0z*(pe5/2);
	prm  pv23 = c0x*(pe6/2)+ c0y*(pe2 + 1) + c0z*(pe4/2);
	prm  pv33 = c0x*(pe5/2)   + c0y*(pe4/2) + c0z*(pe3 + 1);
	'supercell basis vectors in cartesian coords
	prm  sv11 = (t11)*pv11 + (t12)*pv12 + (t13)*pv13;
	prm  sv21 = (t11)*pv21 + (t12)*pv22 + (t13)*pv23;
	prm  sv31 = (t11)*pv31 + (t12)*pv32 + (t13)*pv33;
	prm  sv12 = (t21)*pv11 + (t22)*pv12 + (t23)*pv13;
	prm  sv22 = (t21)*pv21 + (t22)*pv22 + (t23)*pv23;
	prm  sv32 = (t21)*pv31 + (t22)*pv32 + (t23)*pv33;
	prm  sv13 = (t31)*pv11 + (t32)*pv12 + (t33)*pv13;
	prm  sv23 = (t31)*pv21 + (t32)*pv22 + (t33)*pv23;
	prm  sv33 = (t31)*pv31 + (t32)*pv32 + (t33)*pv33;
	'distorted supercell parameters
	prm  sa = Sqrt(sv11^2+sv21^2+sv31^2);
	prm  sb = Sqrt(sv12^2+sv22^2+sv32^2);
	prm  sc = Sqrt(sv13^2+sv23^2+sv33^2);
	prm  salpha = ArcCos((sv12*sv13+sv22*sv23+sv32*sv33)/(sb*sc));
	prm  sbeta  = ArcCos((sv11*sv13+sv21*sv23+sv31*sv33)/(sa*sc));
	prm  sgamma = ArcCos((sv11*sv12+sv21*sv22+sv31*sv32)/(sa*sb));
	prm  sal = salpha*radian;
	prm  sbe = sbeta*radian;
	prm  sga = sgamma*radian;
}


macro NeutronDiffraction
{  
    'NeutronDiffraction macro from ORNL POWGEN User Guidance TOPAS Refinement Examples

    TOF_LAM(0.001)
	neutron_data
	x_calculation_step = Yobs_dx_at(Xo);
	weighting = If(SigmaYobs < 0.0001, 0.001, 1/SigmaYobs^2);     'trouble from very weak diffraction data points
}


macro TOF_x_axis_calibration_POWGEN(d0, d0v, difc, difcv, difa, difav, difb, difbv)
{
    ' on the usage of If_Prm_Eqn_Rpt, see http://topas.dur.ac.uk/unb/forum.php?req=thread&id=630

    #m_argu d0      ' this creates a unique parameter names if d0 starts with @
    #m_argu difc
    #m_argu difa
	#m_argu difb
    If_Prm_Eqn_Rpt(d0, d0v,    ' creates a prm named d0, if d0 starts with a letter or !, and give d0v as its initial value
        min = Val - 15 Yobs_dx_at(.5(X1+X2));
        max = Val + 15 Yobs_dx_at(.5(X1+X2));
        del = .3 Peak_Calculation_Step; )
    If_Prm_Eqn_Rpt(difc, difcv,
        min = Val - 15 Yobs_dx_at(.5(X1+X2));
        max = Val + 15 Yobs_dx_at(.5(X1+X2));
        del .01)
    If_Prm_Eqn_Rpt(difa, difav,
        min = Val - 15 Yobs_dx_at(.5(X1+X2));
        max = Val + 15 Yobs_dx_at(.5(X1+X2));
        del .005)
	If_Prm_Eqn_Rpt(difb, difbv,
        min = Val - 15 Yobs_dx_at(.5(X1+X2));
        max = Val + 15 Yobs_dx_at(.5(X1+X2));
        del .01)
		  	 
    pk_xo = CeV(d0,d0v) + CeV(difc,difcv) D_spacing + CeV(difa,difav) D_spacing^2 + CeV(difb,difbv)/D_spacing;
    'Include if version 6
    'gui_tof_t0 = CeV(d0,d0v);
    'gui_tof_t1 = CeV(difc,difcv);
}


macro GSAS_btb 
{
    'GSAS back to back function type 3, GSAS manual page 150

    push_peak                 
        exp_conv_const = alpha0 + alpha1 / D_spacing + alpha2 / D_spacing^4;
    bring_2nd_peak_to_top   
        exp_conv_const = beta0 + beta1 / D_spacing^4;
    add_pop_1st_2nd_peak 
   
}


macro modified_GSAS_btb 
{
    'From John Evans old peakshape for GEM peak shape 

    push_peak 
        exp_conv_const = alpha0 + alpha1 / D_spacing;
        bring_2nd_peak_to_top
        user_defined_convolution = beta0 Exp( beta1  X) / D_spacing^4;  
                min 0
                max = -Ln(0.001) / beta1;
    add_pop_1st_2nd_peak
     
   
}


macro modified_TOF_Exponential(a0, a0v, a1, a1v, a2, a2v, tpow1, tpow1v, & wexp, & t1, lr)  
	{   
		'TOF exponential 

        #m_argu a0 
		#m_argu a1
		#m_argu a2
	    #m_argu tpow1 
		If_Prm_Eqn_Rpt(a0, a0v, min = Max(Val .3, 1e-6); max = 2 Val + 1; )
		If_Prm_Eqn_Rpt(a1, a1v, min = Max(Val .3, 1e-6); max = 2 Val + 1; )
		If_Prm_Eqn_Rpt(a2, a2v, min = Max(Val .3, 1e-6); max = 2 Val + 1; )
		
		exp_conv_const = lr Constant(t1) / (CeV(a0,a0v) + CeV(a2,a2v)/D_spacing^CeV(tpow1,tpow1v) + CeV(a1,a1v) / D_spacing^ Constant(wexp));
	}


macro GSAS_Ikeda_Carpenter(two_theta, two_thetav, tpow, tpowv, alph0,alph0v,alph1,alph1v,beta0,beta0v,beta1, beta1v, k_eff,k_effv, Acons, Aconsv) 
{ 
    'GSAS type 2 bank, moderator line shape
    'GSAS type_2 function, GSAS manual page 146

    local !two_theta two_thetav 
    local  !lam = 2 D_spacing Sin(0.00872664626 two_theta)  ; 
    prm tpow tpowv min 1 max 5
    prm alph = 1/(alph0 + alph1 lam );
    prm alph0   alph0v min 0.001        
    prm alph1   alph1v min 0.01
    prm beta0   beta0v min 0.01
    prm beta1   beta1v min 0.01 
    prm beta = 1/(beta0 + beta1/lam^2 ); 
    prm k_eff   k_effv min 0.001
    prm Acons   Aconsv  min 0.001
    local  R1 = Acons Exp(-81.799/k_eff lam^2);
    user_defined_convolution =   X^tpow Exp(-X alph);
    min 0 max = 10 /alph;
    'local !R1 = Exp( -950 /(k_eff lam^2)) ;			
        push_peak  
                scale_top_peak = (1-R1); 
        bring_2nd_peak_to_top 
                exp_conv_const = -Ln(0.001) beta; 
				'user_defined_convolution = Exp (-X/beta); 
				min 0 max 1
                scale_top_peak = R1;   
        add_pop_1st_2nd_peak 
 
}


macro wifd_mic_moderator(two_theta,two_thetav,tspow,tspowv,tauf_0,tauf_0v,tauf_1,tauf_1v,tauf_2,tauf_2v,taus_0,taus_0v,taus_1,taus_1v,t_eff,t_effv,hhh,hhhv) 
{   
    'single pulse

    local  !two_theta two_thetav
	prm tspow   tspowv min 1.5 max 5.5   
	prm tauf_0  tauf_0v min 0.001        
	prm tauf_1  tauf_1v min 0.01   
    prm tauf_2  tauf_2v min 0.01	
	prm taus_0  taus_0v min 0.001        
	prm taus_1  taus_1v min 0.001        
	prm t_eff  t_effv min 90.0           
    prm hhh  hhhv min .0001 
    'prm double_pulse  double_pulsev min 0.001
	local  !lam = 2 D_spacing Sin(0.00872664626 two_theta)  ;

 	user_defined_convolution = X^tspow Exp(-X / (tauf_0 + tauf_1 lam + tauf_2 lam^2)); 
    min 0 max = 100 (tauf_0 + tauf_1 lam + tauf_2 lam^2);

	local !storage = Exp( -950 /(t_eff lam^2)) hhh;			
        push_peak  
				scale_top_peak = (1-storage);  'may need to be normalized if rebinning 
        bring_2nd_peak_to_top 
                exp_conv_const = -Ln(0.001) / (taus_0 + taus_1 / lam^2); 
                scale_top_peak = storage;      'may need to be normalized if rebinning
        add_pop_1st_2nd_peak 
}


macro moderator_mic_NOMAD(two_theta,two_thetav,tspow,tspowv,alph0,alph0v,alph1,alph1v,beta0,beta0v,beta1,beta1v,k_eff,k_effv,hk,hkv)       
{
    'modified moderator peak shape, initial from Bill David
    local  !two_theta two_thetav
	prm tspow   tspowv min 1.5 max 5.5   
	prm alph0   alph0v min 0.001        
	prm alph1   alph1v min 0.01         
	prm beta0   beta0v min 0.001        
	prm beta1   beta1v min 0.001        
	prm k_eff   k_effv min 90.0           
    prm hk      hkv min .001 
	local  !lam = 2 D_spacing Sin(0.00872664626 two_theta)  ;
 
 	user_defined_convolution = X^tspow Exp(-X / (alph0 + alph1 lam ));      ' slowdown neutron 1/E
     min 0 max = 10 (alph0 + alph1 lam );
 
	local !Rstorage = Exp(-950/(k_eff lam^2)) hk;	                     ' storage neutron, maxvillian distribution   		
        push_peak  
                scale_top_peak = (1-Rstorage); 
        bring_2nd_peak_to_top 
                exp_conv_const = -Ln(0.001) / (beta0 + beta1 / lam^2); 
                scale_top_peak = Rstorage;   
        add_pop_1st_2nd_peak 
}


macro wifd_pkshape_anneal_1(two_theta,two_thetav,aprm,aprmv,tauf_2,tauf_2v,tauf_1,tauf_1v,tauf_0,tauf_0v,taus_0,taus_0v,taus_1,taus_1v,t_eff,t_effv,hhh,hhhv,hhh_0,hhh_0v) 
{
	local  !two_theta two_thetav
	local  !lam = 2 D_spacing Sin(0.00872664626 two_theta)  ;
	prm aprm    aprmv min 1.5 max 2.5  
	prm tauf_2  tauf_2v min 0.01    val_on_continue = Rand(0.01,1);
	prm tauf_1  tauf_1v min 0.01    val_on_continue = Rand(0.01,1);
	prm tauf_0  tauf_0v min 0.001   val_on_continue = Rand(0.01,1);          
	user_defined_convolution = X^aprm Exp(-X / (tauf_0 + tauf_1 lam + tauf_2 lam^2)); min 0 max = 20 aprm tauf_1 lam;

	prm taus_0  taus_0v min 0.001   val_on_continue = Rand(0.01,1); 
	prm taus_1  taus_1v min 0.001   val_on_continue = Rand(0.01,1);
	prm t_eff  t_effv min 90.0 val_on_continue = Rand(10,1000);
	local storage = Exp( -950 /(t_eff lam lam)) ;			
        push_peak 
                prm hhh  hhhv min .001 
                prm hhh_0  hhh_0v min 0.001
                hat = hhh_0 + hhh D_spacing; 
                scale_top_peak = (1-storage); 
        bring_2nd_peak_to_top 
		  		hat = hhh_0 + hhh D_spacing;
                scale_top_peak = storage;   
                exp_conv_const = -Ln(0.001) (taus_0 + lam taus_1); 
        add_pop_1st_2nd_peak 
}


macro Abs_Lobanov 
{
    'From Pam, Lobanov absorption macro with calculated lambda for wavelength dependence of mu (as per GSAS manual), from Pam Whitfield

    local !Lf = 60.183;      '  90 degree flightpath
    prm   !h_bar 6.626176e-34 ' Plank's constant
    prm   !m_n 1.67495e-27 ' neutron mass
    local !td_const = 10^4 h_bar/(m_n Lf);
    local !sintheta = Sin(Deg_on_2 two_theta);
    local !lambda = Xo 10^4 h_bar/(m_n Lf);
    local !k0 = 1.697653;
    local !k1 = (25.99978 - (0.01911 sintheta^0.5)Exp(-0.024514 sintheta^2)) + 0.109561 sintheta - 26.0456;:  0.01578
    local !k2 = -0.02489 - (0.39499 sintheta^2) + (1.219077 sintheta^3) - (1.31268 sintheta^4) + (0.871081 sintheta^5) 
               - (0.2327 sintheta^6);:0.00535
    local !k3 = 0.003045 + (0.018167 sintheta^2) - (0.03305 sintheta^4);:0.00387
    local !k4 = 1.433902 + (11.07504 sintheta^2) - (8.77629 sintheta^4) + (10.02088 sintheta^6) - (3.36778 sintheta^8);:5.81947
    local !k5 = ((0.013869 - 0.01249 sintheta^2)Exp(3.27094 sintheta^2)) + ((0.337894 + (13.77317 sintheta^2))/((1 + 11.53544 sintheta^2)^1.555039));:0.40847
    local !k6 = (1.933433 / (1 + 23.12967 sintheta^2)^1.686715) - 0.13576 sintheta + 1.163198;:1.09426
    local !k7 = 0.044365 - (0.04259/((1 + 0.41051 sintheta^2)^148.4202));:0.04436
    scale_pks = If( (mu*lambda) < 3, Exp(-k0 (mu*lambda) - k1 (mu*lambda)^2 - k2 (mu*lambda)^3 - k3 (mu*lambda)^4),
				Exp((k4 - k7/(1 + k5 ((mu*lambda)-3))^k6)+k7)
				);				
}


macro POWGEN_instrument    
{
    ' macro using GSAS-style sigmas and gammas for pseudo-Voigt function. Add lines with local parameters to each dataset

    gauss_fwhm = Sqrt((0,sig0 + sig1 * D_spacing^2 + sig2 * D_spacing^4)^2);
    lor_fwhm = Sqrt((0, gam0 + gam1 * D_spacing + gam2 * D_spacing^2)^2); 
    'local sig0 -17.17949_0.47202
    'local sig1  79.48367_0.64650
    'local sig2 -2.15029_0.17308 
    'local gam0 -8.39399_3.15677 
    'local gam1 -23.64600_3.15677
    'local gam2  4.76909_3.15677 
} 


macro POWGEN_instrument_2    
{
    ' macro using GSAS-style sigmas and gammas for pseudo-Voigt function. Add lines with local parameters to each dataset

    gauss_fwhm = Sqrt((sig0 + sig1 * D_spacing^2 + sig2 * D_spacing^4 + sig3/D_spacing^2)^2);
    lor_fwhm = Sqrt((gam0 + gam1 * D_spacing + gam2 * D_spacing^2)^2); 
    'local sig0 -17.17949_0.47202
    'local sig1  79.48367_0.64650
    'local sig2 -2.15029_0.17308 
    'local gam0 -8.39399_3.15677 
    'local gam1 -23.64600_3.15677
    'local gam2  4.76909_3.15677 
    'local sig3  1
}



