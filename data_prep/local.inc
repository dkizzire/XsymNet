' This file is included from TOPAS.INC
' Place local standard macros here


#include cif2.inc
#include pdf.inc
#include cif.inc

'-------------------------------------------------------------------------------------------------------------------------------------
'Johns Macros
'-------------------------------------------------------------------------------------------------------------------------------------
'from Phil Chater Feb 2017
macro SPV_Peak_Type(ha1, hav1, hb1, hbv1, hc1, hcv1, ha2, hav2, hb2, hbv2, hc2, hcv2, lora1, lorav1, lorb1, lorbv1, lorc1, lorcv1, lora2, lorav2, lorb2, lorbv2, lorc2, lorcv2)
{
   #m_argu ha1      #m_argu ha2
   #m_argu hb1      #m_argu hb2
   #m_argu hc1      #m_argu hc2
   #m_argu lora1    #m_argu lora2
   #m_argu lorb1    #m_argu lorb2
   #m_argu lorc1    #m_argu lorc2
   If_Prm_Eqn_Rpt(ha1,   hav1,   min 0.0001 max = 2 Val + .1;, del 0.001)
   If_Prm_Eqn_Rpt(hb1,   hbv1,   min 0.0001 max = 2 Val + .1;, del 0.001)
   If_Prm_Eqn_Rpt(hc1,   hcv1,   min 0.0001 max = 2 Val + .1;, del 0.001)
   If_Prm_Eqn_Rpt(ha2,   hav2,   min 0.0001 max = 2 Val + .1;, del 0.001)
   If_Prm_Eqn_Rpt(hb2,   hbv2,   min 0.0001 max = 2 Val + .1;, del 0.001)
   If_Prm_Eqn_Rpt(hc2,   hcv2,   min 0.0001 max = 2 Val + .1;, del 0.001)
   If_Prm_Eqn_Rpt(lora1, lorav1, min 0.0001 max 1,            del 0.001)
   If_Prm_Eqn_Rpt(lorb1, lorbv1, min 0.0001 max = 2 Val + .1;, del 0.001)
   If_Prm_Eqn_Rpt(lorc1, lorcv1, min 0.0001 max = 2 Val + .1;, del 0.001)
   If_Prm_Eqn_Rpt(lora2, lorav2, min 0.0001 max 1,            del 0.001)
   If_Prm_Eqn_Rpt(lorb2, lorbv2, min 0.0001 max = 2 Val + .1;, del 0.001)
   If_Prm_Eqn_Rpt(lorc2, lorcv2, min 0.0001 max = 2 Val + .1;, del 0.001)
   peak_type spv
      spv_h1 = CeV(ha1,hav1) + CeV(hb1,hbv1) Tan(Th) + CeV(hc1,hcv1) / Cos(Th);
      spv_h2 = CeV(ha2,hav2) + CeV(hb2,hbv2) Tan(Th) + CeV(hc2,hcv2) / Cos(Th);
      spv_l1 = CeV(lora1,lorav1) + CeV(lorb1,lorbv1) Tan(Th) + CeV(lorc1,lorcv1) / Cos(Th);
      spv_l2 = CeV(lora2,lorav2) + CeV(lorb2,lorbv2) Tan(Th) + CeV(lorc2,lorcv2) / Cos(Th);
}
macro SPVII_Peak_Type(ha1, hav1,   hb1, hbv1,   hc1, hcv1,   ma1, mav1,   mb1, mbv1,   mc1, mcv1, ha2, hav2,   hb2, hbv2,   hc2, hcv2,   ma2, mav2,   mb2, mbv2,   mc2, mcv2)
{
   #m_argu ha1  #m_argu ha2
   #m_argu hb1  #m_argu hb2
   #m_argu hc1  #m_argu hc2
   #m_argu ma1  #m_argu ma2
   #m_argu mb1  #m_argu mb2
   #m_argu mc1  #m_argu mc2
   If_Prm_Eqn_Rpt(ha1, hav1, min 0.0001 max 1)    If_Prm_Eqn_Rpt(ha2, hav2, min 0.0001 max 1)
   If_Prm_Eqn_Rpt(hb1, hbv1, min 0.0001 max 1)    If_Prm_Eqn_Rpt(hb2, hbv2, min 0.0001 max 1)
   If_Prm_Eqn_Rpt(hc1, hcv1, min 0.0001 max 1)    If_Prm_Eqn_Rpt(hc2, hcv2, min 0.0001 max 1)
   If_Prm_Eqn_Rpt(ma1, mav1, min 0.0001 max 20)   If_Prm_Eqn_Rpt(ma2, mav2, min 0.0001 max 20)
   If_Prm_Eqn_Rpt(mb1, mbv1, min 0.0001 max 5)    If_Prm_Eqn_Rpt(mb2, mbv2, min 0.0001 max 5)
   If_Prm_Eqn_Rpt(mc1, mcv1, min 0.0001 max 5)    If_Prm_Eqn_Rpt(mc2, mcv2, min 0.0001 max 5) 
   peak_type spvii
      m1 = .6 + CeV(ma1,mav1) + CeV(mb1,mbv1) / Cos(Th) + CeV(mc1,mcv1) / Tan(Th);
      m2 = .6 + CeV(ma2,mav2) + CeV(mb2,mbv2) / Cos(Th) + CeV(mc2,mcv2) / Tan(Th);
      h1 = CeV(ha1,hav1) + CeV(hb1,hbv1) / Cos(Th) + CeV(hc1,hcv1) Tan(Th);
      h2 = CeV(ha2,hav2) + CeV(hb2,hbv2) / Cos(Th) + CeV(hc2,hcv2) Tan(Th);
}

macro StoD_Offset_in2theta(c,v)
{
                #m_argu c
                If_Prm_Eqn_Rpt(c, v, min -1000 max 1000)
                local tth = 2 ArcSin(Lam/(2 D_spacing)); '2theta, in radians
                local radial_offset = CeV(c,v) Tan(tth);
                th2_offset = (ArcTan((Rs Tan(tth) + radial_offset)/Rs)) Rad - tth Rad;
}


macro "\n {\n"}

macro MM_Display_CrystalAxis(mxc, myc, mzc) {
	local = Get(mlx) Get(a); : mxc
	local = Get(mly) Get(b); : myc
	local = Get(mlz) Get(c); : mzc
}



macro MM_Refine_CrystalAxis(mxc, mxv, myc, myv, mzc, mzv, mlx_v, mly_v, mlz_v) {
	#m_argu mxc
	#m_argu myc
	#m_argu mzc
	If_Prm_Eqn_Rpt(mxc, mxv, )
	If_Prm_Eqn_Rpt(myc, myv, )
	If_Prm_Eqn_Rpt(mzc, mzv, )
	mlx = CeV(mxc, mxv) / Get(a); : mlx_v
	mly = CeV(myc, myv) / Get(b); : mly_v 
	mlz = CeV(mzc, mzv) / Get(c); : mlz_v
}	




macro Insert_Peak(p1,v1,p2,v2,p3,v3,p4,v4) {
#m_argu p1 '2theta
#m_argu p2 'lor
#m_argu p3 'fwhm
#m_argu p4 'int
If_Prm_Eqn_Rpt(p1, v1, min =0.9*v1; max =1.1*v1;)
If_Prm_Eqn_Rpt(p2, v2, min .000000000001 max 1)
If_Prm_Eqn_Rpt(p3, v3, min .000000000001 max 1)
If_Prm_Eqn_Rpt(p4, v4, min .000000000001 max 1e9)
xo_Is xo =CeV(p1,v1);  peak_type pv pv_lor =CeV(p2,v2);  pv_fwhm =CeV(p3,v3); I =CeV(p4,v4);
}


macro Inel_Flat_Plate_V2(alpha, alpha_v, sd, sd_v)
	' Version by Matthew Rowles CSIRO Minerals
	' Equations for correction of intensity and sample displacement for flat plate samples
	' in fixed incident beam geometry e.g. when using the Inel CPS120 position sensitive detector
	' Intensity correction derived from Toraya et al. (1993)
	' Sample displacement model derived by Ian Madsen
	' 2Theta = (alpha + beta)
	'   where beta  = angle between diffracted beam(s) and sample surface
	' Rs = 250mm for the Inel CPS120 detector - Rs must be defined in the input file
{
   #m_argu alpha    ' alpha = angle between incident beam and sample surface
   #m_argu sd       ' sd = displacement of the sample (in same units as detector radius)
	If_Prm_Eqn_Rpt(alpha, alpha_v, min 0.0001 max 90)
   If_Prm_Eqn_Rpt(sd, sd_v, min -1 max 1)
 
	scale_pks = 
		IF (2 Th) < CeV(alpha, alpha_v) Deg THEN 
			0                                 
		ELSE
			2 / (1 + (Sin(CeV(alpha, alpha_v) Deg) / Sin(2 Th - CeV(alpha, alpha_v) Deg)))
		ENDIF;
 
	th2_offset = 
		IF CeV(sd, sd_v) == 0 THEN 
			0
		ELSE
			- Rad (CeV(sd, sd_v) Sin(2 Th)) / (Rs Sin(CeV(alpha, alpha_v) Deg))
		ENDIF;
}




'Description: tof peak fitting
'Contributed by: John Evans
'Descritpion: put in a dummy hkl phase to fit individual tof peaks
'prm lor 0.54006`
'prm dsp 51.70683`
'prm !dspsq 0
'tof_peak_fit( @, 2.92093`, @, 2.64961`)

macro tof_peak_fit(p1,v1,p2,v2) {
#m_argu p1
#m_argu p2
If_Prm_Eqn_Rpt(p1, v1, min =0.9*v1; max =1.1*v1;)
If_Prm_Eqn_Rpt(p2, v2, min .000000000001 max 1e9)
hkl_Is phase_name CeV(p1,v1) a =10*CeV(p1,v1); b 0.1 c 0.1 space_group "P1" load hkl_m_d_th2 I { 10 0 0 1 =CeV(p1,v1); 80114.92 =CeV(p2,v2); }
peak_type pv pv_lor =lor; pv_fwhm = dsp*CeV(p1,v1) + dspsq*CeV(p1,v1)^2;
}


'Description: Robust refinement
'Contributed by: Kevin H. Stone, Saul H. Lapidus and Peter W. Stephens
'Reference: J. Appl. Cryst. (2009). 42, 385–391; "Implementation and use of robust refinement in powder diffraction in the presence of impurities"
macro Robust_refinement_xye
{
'Rescale peaks according to robust refinement algorithm
r_exp 4.390
prm test = Get(r_exp); 
prm N = (1/test)^2; 
prm !p0  =0.40007404;
prm !p1  =-2.5949286;
prm !p2  =4.3513542;
prm !p3  =-1.7400101;
prm !p4  =3.6140845 * 10^-1;
prm !p5  =-4.5247609 * 10^-2;
prm !p6  =3.5986364 * 10^-3;
prm !p7  =-1.8328008 * 10^-4;
prm !p8  =5.7937184 * 10^-6;
prm !p9  =-1.035303 * 10^-7;
prm !p10 =7.9903166 * 10^-10;
prm t = ((Yobs - Ycalc)/SigmaYobs);
prm testjohn = If( t < 0.8,
               (1/Max(SigmaYobs^2, 1))*N,
               If(t < 21, N*((((((((((p10*t + p9)*t + p8)*t + p7)
               *t + p6)*t + p5)*t + p4)*t + p3)
               *t + p2)*t + p1)*t +p0)/(Yobs - Ycalc)^2,
               N*(2.0131 * Ln(t) + 3.9183)/(Yobs - Ycalc)^2));
weighting = If( t < 0.8,
              (1/Max(SigmaYobs^2, 1))*N,
              If(t < 21, N*((((((((((p10*t + p9)*t + p8)*t + p7)
              *t + p6)*t + p5)*t + p4)*t + p3)
              *t + p2)*t + p1)*t +p0)/(Yobs - Ycalc)^2,
              N*(2.0131 * Ln(t) + 3.9183)/(Yobs - Ycalc)^2));
recal_weighting_on_iter
}

macro Robust_refinement_poisson
{
'Rescale peaks according to robust refinement algorithm
'N.B. changed SigmaYobs to Yobs^0.5
r_exp 4.390
prm test = Get(r_exp); 
prm N = (1/test)^2; 
prm !p0  =0.40007404;
prm !p1  =-2.5949286;
prm !p2  =4.3513542;
prm !p3  =-1.7400101;
prm !p4  =3.6140845 * 10^-1;
prm !p5  =-4.5247609 * 10^-2;
prm !p6  =3.5986364 * 10^-3;
prm !p7  =-1.8328008 * 10^-4;
prm !p8  =5.7937184 * 10^-6;
prm !p9  =-1.035303 * 10^-7;
prm !p10 =7.9903166 * 10^-10;
prm t = ((Yobs - Ycalc)/Yobs^0.5);
prm testjohn = If( t < 0.8,
               (1/Max(Yobs, 1))*N,
               If(t < 21, N*((((((((((p10*t + p9)*t + p8)*t + p7)
               *t + p6)*t + p5)*t + p4)*t + p3)
               *t + p2)*t + p1)*t +p0)/(Yobs - Ycalc)^2,
               N*(2.0131 * Ln(t) + 3.9183)/(Yobs - Ycalc)^2));
weighting = If( t < 0.8,
              (1/Max(Yobs, 1))*N,
              If(t < 21, N*((((((((((p10*t + p9)*t + p8)*t + p7)
              *t + p6)*t + p5)*t + p4)*t + p3)
              *t + p2)*t + p1)*t +p0)/(Yobs - Ycalc)^2,
              N*(2.0131 * Ln(t) + 3.9183)/(Yobs - Ycalc)^2));
recal_weighting_on_iter
}

macro wifd_pkshape(two_theta,two_thetav,aprm,aprmv,tauf_2,tauf_2v,tauf_1,tauf_1v,tauf_0,tauf_0v,taus_0,taus_0v,taus_1,taus_1v,t_eff,t_effv,hhh,hhhv,hhh_0,hhh_0v) {
	local  !two_theta two_thetav
	local  !lam = 2 D_spacing Sin(0.00872664626 two_theta)  ;
	prm aprm    aprmv min 1.5 max 2.5  
	prm tauf_2  tauf_2v min 0.01
	prm tauf_1  tauf_1v min 0.01 
	prm tauf_0  tauf_0v min 0.001             
	user_defined_convolution = X^aprm Exp(-X / (tauf_0 + tauf_1 lam + tauf_2 lam^2)); min 0 max = 20 aprm tauf_1 lam;

	prm taus_0  taus_0v min 0.001 
	prm taus_1  taus_1v min 0.001 
	prm t_eff  t_effv min 90.0
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

macro wifd_pkshape_anneal(two_theta,two_thetav,aprm,aprmv,tauf_2,tauf_2v,tauf_1,tauf_1v,tauf_0,tauf_0v,taus_0,taus_0v,taus_1,taus_1v,t_eff,t_effv,hhh,hhhv,hhh_0,hhh_0v) {
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

'changed wifd_mic macro to remove /double_pulse jan 2012
macro wifd_mic_new(two_theta,two_thetav,tspow,tspowv,tauf_0,tauf_0v,tauf_1,tauf_1v,taus_0,taus_0v,taus_1,taus_1v,t_eff,t_effv,hhh,hhhv,double_pulse,double_pulsev) {
   local  !two_theta two_thetav
	prm tspow   tspowv min 1.5 max 5.5   
	prm tauf_0  tauf_0v min 0.001        
	prm tauf_1  tauf_1v min 0.01         
	prm taus_0  taus_0v min 0.001        
	prm taus_1  taus_1v min 0.001        
	prm t_eff  t_effv min 90.0           
   prm hhh  hhhv min .001 
   prm double_pulse  double_pulsev min 0.001
	local  !lam = 2 D_spacing Sin(0.00872664626 two_theta)  ;

 	user_defined_convolution = X^tspow Exp(-X / (tauf_0 + tauf_1 lam )); 
   min 0 max = 10 (tauf_0 + tauf_1 lam );

	local !storage = Exp( -950 /(t_eff lam^2)) hhh;			
        push_peak 
                hat = double_pulse ; 
                scale_top_peak = (1-storage); 
        bring_2nd_peak_to_top 
                exp_conv_const = -Ln(0.001) / (taus_0 + taus_1 / lam^2); 
                scale_top_peak = storage;   
        add_pop_1st_2nd_peak 
}

macro wifd_mic_new_anneal(two_theta,two_thetav,tspow,tspowv,tauf_0,tauf_0v,tauf_1,tauf_1v,taus_0,taus_0v,taus_1,taus_1v,t_eff,t_effv,hhh,hhhv,double_pulse,double_pulsev) {
   local  !two_theta two_thetav
	prm tspow   tspowv min 1.5 max 5.5            val_on_continue = Val * Rand(0.8,1.2);   
	prm tauf_0  tauf_0v min 0.001                 val_on_continue = Val * Rand(0.8,1.2);
	prm tauf_1  tauf_1v min 0.01                  val_on_continue = Val * Rand(0.8,1.2);
	prm taus_0  taus_0v min 0.001                 val_on_continue = Val * Rand(0.8,1.2);
	prm taus_1  taus_1v min 0.001                 val_on_continue = Val * Rand(0.8,1.2);
	prm t_eff  t_effv min 90.0                    val_on_continue = Val * Rand(0.8,1.2);
   prm hhh  hhhv min .001                        val_on_continue = Val * Rand(0.8,1.2);
   prm double_pulse  double_pulsev min 0.001     val_on_continue = Val * Rand(0.8,1.2);
	local  !lam = 2 D_spacing Sin(0.00872664626 two_theta)  ;

 	user_defined_convolution = X^tspow Exp(-X / (tauf_0 + tauf_1 lam )); 
   min 0 max = 10 (tauf_0 + tauf_1 lam );

	local !storage = Exp( -950 /(t_eff lam^2)) hhh;			
        push_peak 
                hat = double_pulse ; 
                scale_top_peak = (1-storage); 
        bring_2nd_peak_to_top 
                exp_conv_const = -Ln(0.001) / (taus_0 + taus_1 / lam^2); 
                scale_top_peak = storage;   
        add_pop_1st_2nd_peak 
}




macro wifd_mic(two_theta,two_thetav,tspow,tspowv,tauf_0,tauf_0v,tauf_1,tauf_1v,taus_0,taus_0v,taus_1,taus_1v,t_eff,t_effv,hhh,hhhv,double_pulse,double_pulsev) {
   local  !two_theta two_thetav
	prm tspow   tspowv min 1.5 max 5.5   
	prm tauf_0  tauf_0v min 0.001        
	prm tauf_1  tauf_1v min 0.01         
	prm taus_0  taus_0v min 0.001        
	prm taus_1  taus_1v min 0.001        
	prm t_eff  t_effv min 90.0           
   prm hhh  hhhv min .001 
   prm double_pulse  double_pulsev min 0.001
	local  !lam = 2 D_spacing Sin(0.00872664626 two_theta)  ;

 	user_defined_convolution = X^tspow Exp(-X / (tauf_0 + tauf_1 lam )); 
   min 0 max = 10 (tauf_0 + tauf_1 lam );

	local !storage = Exp( -950 /(t_eff lam^2)) hhh;			
        push_peak 
                hat = double_pulse ; 
                scale_top_peak = (1-storage)/double_pulse; 
        bring_2nd_peak_to_top 
                exp_conv_const = -Ln(0.001) / (taus_0 + taus_1 / lam^2); 
                scale_top_peak = storage/double_pulse;   
        add_pop_1st_2nd_peak 
}



'Ken's cif output macros
' NB This replaces the default Out_FCF macro in topas.inc
'    in order to include the crucial "_shelx_refln_list_code 4"



macro Ken_Out_FCF(file)
{
   out file
      Out_String("\ndata_")
      Out_String("\n_shelx_refln_list_code 4")
      Out(Get(a), "\n_cell_length_a %V")
      Out(Get(b), "\n_cell_length_b %V")
      Out(Get(c), "\n_cell_length_c %V")
      Out(Get(al), "\n_cell_angle_alpha %V")
      Out(Get(be), "\n_cell_angle_beta  %V")
      Out(Get(ga), "\n_cell_angle_gamma %V")
      Out_String("\n_shelx_F_squared_multiplier 1")
 
      Out_String("\nloop_\n_symmetry_equiv_pos_as_xyz")
      Out(Get(sp_xyzs_txt),  "%s")
 
      Out_String("\nloop_")
      Out_String("\n_refln_index_h")
      Out_String("\n_refln_index_k")
      Out_String("\n_refln_index_l")
      Out_String("\n_refln_F_squared_calc")
      Out_String("\n_refln_F_squared_meas")
      Out_String("\n_refln_F_squared_sigma")
      Out_String("\n_refln_observed_status")
      phase_out file append
         load out_record out_fmt out_eqn
         {
            "\n%4.0f" = H;
            "%4.0f" = K;
            "%4.0f" = L;
            "%12.2f" = I_no_scale_pks;
            "%12.2f" = Iobs_no_scale_pks;
            "%10.2f o" = Iobs_no_scale_pks_err;
         }
}
 
macro Out_PowderDataBlock(file)
{
   out file
   Out_String("loop_")
   Out_String("\n_pd_meas_2theta_scan")
   Out_String("\n_pd_proc_intensity_total")
   Out_String("\n_pd_calc_intensity_total")
   Out_String("\n_pd_proc_ls_weight")
   Out_String("\n")      
   xdd_out file append load out_record out_fmt out_eqn
   {
       " %11.5f  " = X;
       " %11.5f  " = Yobs;
       " %11.5f  " = Ycalc;
       " %11.5f\n" = 1./(SigmaYobs*SigmaYobs);
   }
}

macro Out_X_Yobs_Ycalc_DiffOverESD(file)
{
   xdd_out file load out_record out_fmt out_eqn
   {
       " %11.5f  " = X;
       " %11.5f  " = Yobs;
       " %11.5f  " = Ycalc;
       " %11.5f\n" = (Yobs-Ycalc)/SigmaYobs;
   }
}
 
macro Out_Tick(file)
{
   phase_out file load out_record out_fmt out_eqn
   {
      " %11.5f -200\n"   = 2 Rad Th;
   }
}


'put coordinates out in convenient min/max format
macro Out_min_max(file)
{
   out file
      Out_String("\nmacro A1(param,val,val2) {x param val min = val2 - 0.1; max = val2 + 0.1;  val_on_continue = val2;}")
      Out_String("\nmacro A2(param,val,val2) {y param val min = val2 - 0.1; max = val2 + 0.1;  val_on_continue = val2;}")
      Out_String("\nmacro A3(param,val,val2) {z param val min = val2 - 0.1; max = val2 + 0.1;  val_on_continue = val2;}")

         atom_out file append
            load out_record out_fmt out_eqn
            {
               "\n site %5s " = Get_From_String(Get(current_atom), site);
					" A1( x%s" = Get_From_String(Get(current_atom), site);
					",%8.5f" = Get_From_String(Get(current_atom), x);
					" ,%8.5f" = Get_From_String(Get(current_atom), x);
               ") A2( y%s" = Get_From_String(Get(current_atom), site);
					",%8.5f" = Get_From_String(Get(current_atom), y);
					" ,%8.5f" = Get_From_String(Get(current_atom), y);
               ") A3( z%s" = Get_From_String(Get(current_atom), site);
					",%8.5f" = Get_From_String(Get(current_atom), z);
					" ,%8.5f" = Get_From_String(Get(current_atom), z);
               ") occ %5s" = Get_From_String(Get(current_atom), atom);
               " %6.4f" = Get_From_String(Get(current_atom), occ);
               " beq bvalue%s" = Get_From_String(Get(current_atom), site);
               " %6.4f" = Get_From_String(Get(current_atom), beq);
            }
}  

macro Out_fold_662(file)
{
   out file
   		prm afold = na;
         atom_out file 
            load out_record out_fmt out_eqn
            {
               "\n site %s " = Get_From_String(Get(current_atom), site);
					" x = Mod(x%s" = Get_From_String(Get(current_atom), site);
					"*6,1); y= Mod(y%s" = Get_From_String(Get(current_atom), site);
					"*6,1); z= Mod(z%s" = Get_From_String(Get(current_atom), site);
               "*2,1); occ %s" = Get_From_String(Get(current_atom), atom);
               " = %3.2f" = Get_From_String(Get(current_atom), occ);
               "/72;" = Get_From_String(Get(current_atom), site);
               " beq %6.4f" = Get_From_String(Get(current_atom), beq);
            }
} 		


macro Out_Chloe_cloud(file)
{
   out file
         atom_out file append
            load out_record out_fmt out_eqn
           {
               "\n%s" = Get_From_String(Get(current_atom), site);
					" %V" = Get_From_String(Get(current_atom), x);
               " %V" = Get_From_String(Get(current_atom), y);
               " %V" = Get_From_String(Get(current_atom), z);
            }
} 

'output cif adps for crystalmaker
'macro removes a couple of default topas cif items that crystalmaker doesn't like reading
macro Out_CIF_crystalmaker(file) {Out_CIF_crystalmaker_1(file) Out_CIF_crystalmaker_2(file)}
macro Out_CIF_crystalmaker_1(file)
{
   out file
      Out_String("\ndata_")
      Out(Get(phase_name), "\n_chemical_name_mineral ?%s?")
      Out(Get(a), "\n_cell_length_a  %V")
      Out(Get(b), "\n_cell_length_b  %V")
      Out(Get(c), "\n_cell_length_c  %V")
      Out(Get(al), "\n_cell_angle_alpha %V")
      Out(Get(be), "\n_cell_angle_beta  %V")
      Out(Get(ga), "\n_cell_angle_gamma %V")
      Out(Get(cell_volume), "\n_cell_volume %V")
		
      Out(Get(sp_grp_char), "\n_space_group %s")

      Out_String("\nloop_\n_symmetry_equiv_pos_as_xyz")
      Out(Get(sp_xyzs_txt),  "%s")

		Out_String("\nloop_")
         Out_String("\n_atom_site_label")
'         Out_String("\n_atom_site_type_symbol")
'         Out_String("\n_atom_site_symmetry_multiplicity")
         Out_String("\n_atom_site_fract_x")
         Out_String("\n_atom_site_fract_y")
         Out_String("\n_atom_site_fract_z")
         Out_String("\n_atom_site_occupancy")
         Out_String("\n_atom_site_B_iso_or_equiv")
         atom_out file append
            load out_record out_fmt out_eqn
            {
               "\n%s" = Get_From_String(Get(current_atom), site);
'					" %s" = Get_From_String(Get(current_atom), atom);
'               " %3.0f" = Get_From_String(Get(current_atom), num_posns);
					" %V" = Get_From_String(Get(current_atom), x);
               " %V" = Get_From_String(Get(current_atom), y);
               " %V" = Get_From_String(Get(current_atom), z);
               " %V" = Get_From_String(Get(current_atom), occ);
               " %V" = Get_From_String(Get(current_atom), beq);
            }
}

macro Out_CIF_crystalmaker_2(file)
{
   out file append
      Out_String("\nloop_")
      Out_String("\n_atom_site_aniso_label")
'      Out_String("\n_atom_site_type_symbol")
      Out_String("\n_atom_site_aniso_U_11")
      Out_String("\n_atom_site_aniso_U_22")
      Out_String("\n_atom_site_aniso_U_33")
      Out_String("\n_atom_site_aniso_U_12")
      Out_String("\n_atom_site_aniso_U_13")
      Out_String("\n_atom_site_aniso_U_23")
      atom_out file append
         load out_record out_fmt out_eqn
         {
            "\n%s" = Get_From_String(Get(current_atom), site);
'            " %s" = Get_From_String(Get(current_atom), atom);
            " %V" = Get_From_String(Get(current_atom), u11);
            " %V" = Get_From_String(Get(current_atom), u22);
            " %V" = Get_From_String(Get(current_atom), u33);
            " %V" = Get_From_String(Get(current_atom), u12);
            " %V" = Get_From_String(Get(current_atom), u13);
            " %V" = Get_From_String(Get(current_atom), u23);
          }
}

'output an xtl file
macro Out_xtl(file) {
	out file
	Out_String("TITLE ")
	Out(Get(phase_name))
	Out_String("\nCELL\n")
	Out(Get(a), "%9.4f")
   Out(Get(b), "%9.4f")
   Out(Get(c), "%9.4f")
   Out(Get(al), "%9.4f")
   Out(Get(be), "%9.4f")
   Out(Get(ga), "%9.4f")
   Out_String("\nATOMS")
   Out_String("\nNAME X Y Z")
   atom_out file append
   load out_record out_fmt out_eqn
   {
      "\n%-4s" = Get_From_String(Get(current_atom), site);
      " %11.8f " = Get_From_String(Get(current_atom), x);
      " %11.8f " = Get_From_String(Get(current_atom), y);
      " %11.8f " = Get_From_String(Get(current_atom), z);
   }
   out file append
   Out_String("\nEOF")
}


'generate dif file in uxd format
macro Out_Dif(file)
{
   out file
      Out_String("_D-I\n")
   phase_out file append load out_record out_fmt out_eqn
   {
     " %11.5f"   = D_spacing;
      " %11.5f\n" = I_after_scale_pks;
   }
}		


'gnuplot hkl labels
macro Out_gnuplot_labels(file)

{
	   out file
      Out_String("# put quotation marks round hkl labels; change yoffset value to move labels\n")
      Out_String("set nolabel\n")
      Out_String("yoffset=1\n")
   phase_out file append load out_record out_fmt out_eqn
   {
     "set label  %3.0f" = H; " %3.0f" =K; " %3.0f" = L;
	  " at %11.5f" = 2 57.2957795130823 Th;
      ", %11.5f * yoffset \n" = I_after_scale_pks;
   }
	
}

macro Out_gnuplot_tics(file)
{
   phase_out file append load out_record out_fmt out_eqn
   {
     " %11.5f"   = 2 57.2957795130823 Th;
      " %11.5f\n" = I_after_scale_pks;
   }
}		


'stop adps having to be positive definite in v4

      macro adp_no_limits
{
	move_to u11 min = Val - .05; max = Val + .05;
	move_to u22 min = Val - .05; max = Val + .05;
	move_to u33 min = Val - .05; max = Val + .05;
	move_to u12 min = Val - .05; max = Val + .05;
	move_to u13 min = Val - .05; max = Val + .05;
	move_to u23 min = Val - .05; max = Val + .05;
}



'anneal a tchz function

macro calc_pattern(start,stop,step,filename) {yobs_eqn !aac.xy = 1; min start max stop del step
xdd_out filename load out_record out_fmt out_eqn
   {
       " %11.5f  " = X;
       " %11.5f\n" = Ycalc;
   } 
}




macro TCHZ_Peak_Type_Anneal(u, uv, v, vv, w, wv, z, zv, x, xv, y, yv)
{
   #m_argu u
   #m_argu v
   #m_argu w
   #m_argu z
   #m_argu x
   #m_argu y
   If_Prm_Eqn_Rpt(u, uv, min = Max(-1, Val-.1); max = Min(2, Val+.1); del 1.0e-4)            val_on_continue = Rand(0.001, 0.01);
   If_Prm_Eqn_Rpt(v, vv, min = Max(-1, Val-.1); max = Min(2, Val+.1); del 1.0e-4)            val_on_continue = Rand(0.001, 0.01);
   If_Prm_Eqn_Rpt(w, wv, min = Max(-1, Val-.1); max = Min(2, Val+.1); del 1.0e-4)            val_on_continue = Rand(0.001, 0.01);
   If_Prm_Eqn_Rpt(z, zv, min = Max(-1, Val-.1); max = Min(2, Val+.1); del 1.0e-4)            val_on_continue = Rand(0.001, 0.01);
   If_Prm_Eqn_Rpt(x, xv, min = Max(0.0001, Val-.1); max = Min(2, Val+.1); del 1.0e-4 )       val_on_continue = Rand(0.001, 0.01);
   If_Prm_Eqn_Rpt(y, yv, min = Max(0.0001, Val-.1); max = Min(2, Val+.1); del 1.0e-4 )       val_on_continue = Rand(0.001, 0.01);
	TCHZ_Peak_Type(CeV(u, uv), CeV(v, vv), CeV(w, wv), CeV(z, zv), CeV(x, xv), CeV(y, yv))		
}






'creates a shelx type hkl file 22/2/05

macro Out_shelx_f2(file)
{
	out file
	phase_out file 
	        load out_record out_fmt out_eqn
	{
            "%4.0f" = H;
            "%4.0f" = K;
            "%4.0f" = L;
            "%8.0f" = ((B01+A11)^2 + (A01-B11)^2);
            "%8.0f\n" = ((B01+A11)^2 + (A01-B11)^2)^0.5;
	    }
}	    

macro Get_Distance(sites, c, d_calc)
{
   #m_ifarg c ""
      #m_unique_not_refine c
   #m_endif
	sites_distance c 
		load site_to_restrain { sites }
	prm = c; : d_calc

}

 macro Get_Angle(sites, c, ang_calc)
 {
   #m_ifarg c ""
      #m_unique_not_refine c
   #m_endif
	sites_geometry s3
		load site_to_restrain { sites }
	prm   c = Sites_Geometry_Angle(s3); : ang_calc
  }  

macro Rietveld_Tics(filename) { Rietveld_Tics(filename,1,0) }
macro Rietveld_Tics(filename,difc_val) { Rietveld_Tics(filename,difc_val,0) }
macro Rietveld_Tics(filename,difc_val,yposn)
{
      phase_out filename load out_record out_fmt out_eqn   {
       	"%14.5f" = D_spacing * difc_val; 
        " %9.2f"  = yposn;
       	" %3.0f" = H; 
         " %3.0f" = K;
         " %3.0f\n" = L;
      }  
}
 
macro Rietveld_Tics_2th(filename) { Rietveld_Tics_2th(filename,0) }
macro Rietveld_Tics_2th(filename,yposn)
{
      phase_out filename load out_record out_fmt out_eqn   {
       	"%14.5f" = 2* ArcSin(Lam/(2*D_spacing)) * (360/(2*Pi)); 
        " %9.2f"  = yposn;
       	" %3.0f" = H; 
         " %3.0f" = K;
         " %3.0f\n" = L;
      }  
}

macro Rietveld_Plot(filename)
{
xdd_out filename load out_record out_fmt out_eqn
   {
       " %11.5f  " = X;
       " %11.5f  " = Yobs;
       " %11.5f  " = Ycalc;
       " %11.5f\n" = Yobs-Ycalc;
   } 
}

macro Rietveld_Plot_Root(filename)
{
xdd_out filename load out_record out_fmt out_eqn
   {
       " %11.5f  " = X;
       " %11.5f  " = Yobs^0.5;
       " %11.5f  " = Ycalc^0.5;
       " %11.5f\n" = Yobs^0.5-Ycalc^0.5;
   } 
}

macro Rietveld_Plot_tof_dspacing(filename,difc_val)
{
xdd_out filename load out_record out_fmt out_eqn
   {
       " %11.5f  " = X/difc_val;
       " %11.5f  " = Yobs;
       " %11.5f  " = Ycalc;
       " %11.5f\n" = Yobs-Ycalc;
   } 
}

macro Rietveld_Plot_2th_dspacing(filename)
{
xdd_out filename load out_record out_fmt out_eqn
   {
       " %11.5f  " = Lam/(2 * Sin(X Deg/2));
       " %11.5f  " = Yobs;
       " %11.5f  " = Ycalc;
       " %11.5f\n" = Yobs-Ycalc;
   } 
}



macro write_atoms_header(file) {
out file append
Out_String("\nTITL output from topas")
Out(Get(sp_grp_char), "\nSPGP %s")
Out(Get(a), "\nCELL  %f")
Out(Get(b), "  %f")
Out(Get(c), "  %f")
Out(Get(al), " %f")
Out(Get(be), "  %f")
Out(Get(ga), " %f")
Out_String("\nFIELDS LAB COO TFU")
}

	macro Out_Site
	{
		load out_record out_fmt out_eqn {
			 "\n%-7s" =Get(site); ' Example of left justifing the site name into a string of 7 characters wide using the '-' sign
			 " %9.5f" =Get(x);
			 " %9.5f" =Get(y);
			 " %9.5f" =Get(z);
}
		}


macro Monochromator_Fix_Biso(p1,v1,p2,v2)
{
prm p1 v1 min = Max(.0001, Val/2-1); max = Min(90, Val 2+1);
prm p2 v2 min = -10; max = 10; 
scale_pks = (Exp(-2*p2 Sin(Th)^2/Lam^2)) (1 + Cos((p1) 0.01745329251994)^2 Cos(2 Th)^2) / (Sin(Th)^2 Cos(Th));
}

macro Fix_Biso(p2,v2)
{
prm p2 v2 min = -10; max =10; 
scale_pks = (Exp(-2*p2 Sin(Th)^2/Lam^2)) ;
}

macro TOF_XYE_JOHN(path, calc_step)
{
	xdd path xye_format
		neutron_data       
   	x_calculation_step calc_step
   	weighting = 1 / SigmaYobs^2;
}  

macro write_atoms(file)
{

out file 'append
Out_String("\nTITL output from topas")
Out(Get(sp_grp_char), "\nSPGP %s")
Out(Get(a), "\nCELL  %f")
Out(Get(b), "  %f")
Out(Get(c), "  %f")
Out(Get(al), " %f")
Out(Get(be), "  %f")
Out(Get(ga), " %f")
Out_String("\nFIELDS LAB COO TFB")
         atom_out file append
            load out_record out_fmt out_eqn
            {
               "\n%-7s" = Get_From_String(Get(current_atom), site);
               " %9.5f" = Get_From_String(Get(current_atom), x);
               " %9.5f" = Get_From_String(Get(current_atom), y);
               " %9.5f" = Get_From_String(Get(current_atom), z);
'               " %9.5f" = Get_From_String(Get(current_atom), occ);
               " \n%9.5f" = Get_From_String(Get(current_atom), beq);
            }
}

'just write coords in atoms format
macro append_atoms(file)
{
         atom_out file append
            load out_record out_fmt out_eqn
            {
               "\n%-7s" = Get_From_String(Get(current_atom), site);
               " %9.5f" = Get_From_String(Get(current_atom), x);
               " %9.5f" = Get_From_String(Get(current_atom), y);
               " %9.5f" = Get_From_String(Get(current_atom), z);
               "\n%9.5f" = Get_From_String(Get(current_atom), beq);
            }
}

macro write_cell(file)
{
out file 'append
Out(Get(r_wp), "  %11.5f")
Out(Get(a), "  %V")
Out(Get(b), "  %V")
Out(Get(c), "  %V")
Out(Get(al), " %V")
Out(Get(be), "  %V")
Out(Get(ga), " %V")
            }



macro write_atoms_adp(file)
{

out file 'append
Out_String("\nTITL output from topas")
Out(Get(sp_grp_char), "\nSPGP %s")
Out(Get(a), "\nCELL  %f")
Out(Get(b), "  %f")
Out(Get(c), "  %f")
Out(Get(al), " %f")
Out(Get(be), "  %f")
Out(Get(ga), " %f")
Out_String("\nFIELDS LAB COO TFU")
         atom_out file append
            load out_record out_fmt out_eqn
            {
               "\n%-7s" = Get_From_String(Get(current_atom), site);
               " %9.5f" = Get_From_String(Get(current_atom), x);
               " %9.5f" = Get_From_String(Get(current_atom), y);
               " %9.5f" = Get_From_String(Get(current_atom), z);
'               " %9.5f" = Get_From_String(Get(current_atom), occ);
               " \n%9.5f" = Get_From_String(Get(current_atom), u11);
               " %9.5f" = Get_From_String(Get(current_atom), u22);
               " %9.5f" = Get_From_String(Get(current_atom), u33);
               " %9.5f" = Get_From_String(Get(current_atom), u12);
               " %9.5f" = Get_From_String(Get(current_atom), u13);
               " %9.5f" = Get_From_String(Get(current_atom), u23);
            }
}

'simple macro to output name, x, y, z

macro write_coords(file)
{
         atom_out file 
            load out_record out_fmt out_eqn
            {
               "%-7s" = Get_From_String(Get(current_atom), site);
               " %9.5f" = Get_From_String(Get(current_atom), x);
               " %9.5f" = Get_From_String(Get(current_atom), y);
               " %9.5f" = Get_From_String(Get(current_atom), z);
               " %9.5f\n" = Get_From_String(Get(current_atom), beq);
           }
}

macro append_coords(file)
{
         atom_out file append
            load out_record out_fmt out_eqn
            {
               "%-7s" = Get_From_String(Get(current_atom), site);
               " %9.5f" = Get_From_String(Get(current_atom), x);
               " %9.5f" = Get_From_String(Get(current_atom), y);
               " %9.5f" = Get_From_String(Get(current_atom), z);
               " %9.5f\n" = Get_From_String(Get(current_atom), beq);
           }
}

macro write_sites(file)
{
   out file
Out(te, "\n prm !te %11.5f")
Out(Get(sp_grp_char), "\nspace_group %s")
Out(Get(a), "\n a  %f")
Out(Get(b), " b  %f")
Out(Get(c), " c  %f")
Out(Get(al), " al %f")
Out(Get(be), " be  %f")
Out(Get(ga), " ga %f")
         atom_out file append
            load out_record out_fmt out_eqn
            {
               "\n site %s" = Get_From_String(Get(current_atom), site);
               " x %V" = Get_From_String(Get(current_atom), x);
               " y %V" = Get_From_String(Get(current_atom), y);
               " z %V occ " = Get_From_String(Get(current_atom), z);
               " %s" = Get_From_String(Get(current_atom), atom);
               "  %V" = Get_From_String(Get(current_atom), occ);
               " beq %V" = Get_From_String(Get(current_atom), beq);
            }
}



macro Out_CIF_STR_APPEND(file)
{
   out file append
   Out(te, "\n\n->Cif coordinates for temperature:  %11.5f")
      Out_String("\ndata_")
      Out(Get(phase_name), "\n_chemical_name_mineral ¢%s¢")
      Out(Get(a), "\n_cell_length_a  %V")
      Out(Get(b), "\n_cell_length_b  %V")
      Out(Get(c), "\n_cell_length_c  %V")
      Out(Get(al), "\n_cell_angle_alpha %V")
      Out(Get(be), "\n_cell_angle_beta  %V")
      Out(Get(ga), "\n_cell_angle_gamma %V")
      Out(Get(volume), "\n_cell_volume %V")

      Out_String("\nloop_\n_symmetry_equiv_pos_as_xyz")
      Out(Get(sp_xyzs_txt),  "%s")

      Out_String("\nloop_")
         Out_String("\n_atom_site_label")
         Out_String("\n_atom_site_type_symbol")
         Out_String("\n_atom_site_symmetry_multiplicity")
         Out_String("\n_atom_site_fract_x")
         Out_String("\n_atom_site_fract_y")
         Out_String("\n_atom_site_fract_z")
         Out_String("\n_atom_site_occupancy")
         Out_String("\n_atom_site_B_iso_or_equiv")
         atom_out file append
            load out_record out_fmt out_eqn
            {
               "\n%s" = Get_From_String(Get(current_atom), site);
               " %s" = Get_From_String(Get(current_atom), atom);
               " %3.0f" = Get_From_String(Get(current_atom), num_posns);
               " %V" = Get_From_String(Get(current_atom), x);
               " %V" = Get_From_String(Get(current_atom), y);
               " %V" = Get_From_String(Get(current_atom), z);
               " %V" = Get_From_String(Get(current_atom), occ);
               " %V" = Get_From_String(Get(current_atom), beq);
            }
}

macro gem_instrumental_peakshape( ,prma0, ,prma1, ,prmb0, ,prmb1)
{
   push_peak        
      exp_conv_const = prma0 + prma1 / D_spacing;
	bring_2nd_peak_to_top
      user_defined_convolution        
          = prmb0 Exp(- prmb1  X) / D_spacing^4;  
            min 0
            max = -Ln(0.001) / prmb1;
   add_pop_1st_2nd_peak
      peak_buffer_step 1 'report_on

}

macro tof_sample_peakshape(lor,lor_val,dsp,dsp_val,dspsq,dspsq_val) 
{         
      prm dsp dsp_val del = 0.05 Val + 1; min 1
      prm dspsq dspsq_val del = 0.05 Val + 1; min -5
      peak_type pv             
         pv_lor lor lor_val
         pv_fwhm  = dsp D_spacing + dspsq D_spacing^2;    
}

macro H_ride(h_site, c_site, xv, yv, zv, beqcv)
{
	site h_site
		x = Constant((xv)-Get(Find_Child(Get(site_recs), c_site), x)) + Get(Find_Child(Get(site_recs), c_site), x);
		y = Constant((yv)-Get(Find_Child(Get(site_recs), c_site), y)) + Get(Find_Child(Get(site_recs), c_site), y);
		z = Constant((zv)-Get(Find_Child(Get(site_recs), c_site), z)) + Get(Find_Child(Get(site_recs), c_site), z);
		occ H 1
		beq beqcv
}
'-------------------------------------------------------------------------------------------------------------------------------------

'Graham's macros
macro e0_from_Strain_write(e0, sgc, sgv, slc, slv)
{
   #m_argu sgc
   #m_argu slc
   prm = Voigt_FWHM_GL(CeV_or_0(sgc, sgv), CeV_or_0(slc, slv)) .25; : e0
   #m_ifarg sgv "" #m_else
      Strain_G(sgc, sgv)
   #m_endif
   #m_ifarg slv "" #m_else
      Strain_L(slc, slv)
   #m_endif
}

'Alan's macro
macro Angle_Distance_Restrain4(c, t, t_calc, tol, wscale)
{
	prm = c; : t_calc
   penalty = 
IF Cycle_Iter < 13 
THEN
	(wscale)   
IF c < ((t)-(tol)) 
THEN    
(c - ((t) - (tol)) )^2
ELSE
IF c > ((t)+(tol)) 
THEN
(c - ((t) + (tol)))^2
ELSE
0
ENDIF
ENDIF
ELSE 0
ENDIF
; 
}
macro Distance_Restrain4(sites, c, t, t_calc, tol, wscale)
{
   #m_ifarg c ""
      #m_unique_not_refine c
   #m_endif
	sites_distance c 
		load site_to_restrain { sites }
	Angle_Distance_Restrain4(c, t, t_calc, tol, wscale)
}
macro Distance_Restrain4(sites, t, t_calc, tol, wscale)
{
	Distance_Restrain4(sites,, t, t_calc, tol, wscale)	
}

'From Alec McLennan Liverpool 28/2/2013

macro CIF_Companion(phase_numberval, Z)
{
local !phase_number phase_numberval
local !cell_formula_units_Z Z
}

macro LP_Factor1(LP_Factorv) { LP_Factor1(,LP_Factorv) }
macro LP_Factor1(LP_Factorval, LP_Factorv)
{
   local !LP_Factorval LP_Factorv
   If_Prm_Eqn_Rpt(LP_Factorval, LP_Factorv, min .0001 max 90)
	scale_pks = (1 + Cos(CeV(LP_Factorval,LP_Factorv) Deg)^2 Cos(2 Th)^2) / (Sin(Th)^2 Cos(Th));
}
macro Out_General_CIF_test(file)
{
 	out file
 		if Prm_There(cell_formula_units_Z) {
 		} else {
 		local !cell_formula_units_Z 1
 		}
 		if Prm_There(phase_number) {
 		} else {
 		local !phase_number 1
 		}
   	Out_String("data_overall")
   	Out(Abs(Get(refine_ls_shift_on_su_max)), "\n_refine_ls_shift/su_max %3.4f")       
      Out_String("\n_computing_structure_refinement TOPAS-V5")
      Out(Get(number_of_parameters), "\n_refine_ls_number_parameters %g")  	
      if Obj_There(scrysts) {  
      Out((Get(gof)^2), "\n_refine_ls_goodness_of_fit_gt %1.5f")  
      ' / Max((Sqrt(Yobs)),1)
      xdd_sum !denominator = Abs(Sqrt(Yobs));
      xdd_sum !numerator = Abs(Abs(Sqrt(Yobs)) - Abs(Sqrt(Ycalc)));
      prm Unweighted_r_factor_based_F = numerator/ denominator;
      Out(Unweighted_r_factor_based_F, "\n_refine_ls_R_factor_gt %1.5f")  
      Out(Get(r_wp), "\n_refine_ls_wR_factor_gt %1.5f")
      } else {
      Out_String("\n_pd_block_id \n")  
   	for xdds {
   	Out(Get(xdd_path_name), "%s|") 
   	}
   	Out_String("overall") 
      Out((Get(gof)^2), "\n_refine_ls_goodness_of_fit_all %1.5f")  
      Out(Get(r_p), "\n_pd_proc_ls_prof_R_factor %1.5f")
      Out(Get(r_wp), "\n_pd_proc_ls_prof_wR_factor %1.5f")
      Out(Get(r_exp), "\n_pd_proc_ls_prof_wR_expected %1.5f")   	 
      Out_String("\nloop_ _pd_phase_block_id")
      for xdds {
      for strs {
    	Out(phase_number, "\nphase_%V")
    	Out(Get(sp_grp_char), "_%s")
    	}
    	}
     Out_String("\nloop_ _pd_block_diffractogram_id\n")	    
     for xdds {
     Out(Get(xdd_path_name), "%s\n")
     }
     }
     Out_String("\n#######End of overall data loop#######")

if Obj_There(scrysts) {   
     	Out_CIFSTR(file)
		Out_FCFSXtal(file)
} else {
    
      	Out_CIFSTR(file)
for xdds {
			Out_CIFdata(file)
		}
		      
}
}
macro Out_CIFSTR(file) {
out file append
for strs {
			if Obj_There(scrysts) {
			Out_String("\n#######Start of single crystal structure loop#######")
			} else {
			Out_String("\n#######Start of powder structure loop#######")
 			Out(phase_number, "\ndata_phase_%V")     
			Out(phase_number, "\n_pd_block_id phase_%V")
			Out(Get(sp_grp_char), "_%s")
			Out_String("\n_cell_measurement_temperature")
			if Prm_There(Temperature) {
    		Out(Get(Temperature), " %V")
    		} else {
    		Out_String(" 298.15")
    		}    		
      	Out(phase_number, "\n_pd_phase_id %V")
			Out_String("\n_pd_phase_mass_%")
      	Out(Get(weight_percent), " %V")   
      	}
    		
      	Out(Get(a), "\n_cell_length_a %V")
      	Out(Get(b), "\n_cell_length_b %V")
      	Out(Get(c), "\n_cell_length_c %V")
      	Out(Get(al), "\n_cell_angle_alpha %V")
      	Out(Get(be), "\n_cell_angle_beta %V")
      	Out(Get(ga), "\n_cell_angle_gamma %V")
      	Out(Get(cell_volume), "\n_cell_volume %V")      	           	
      	if Obj_There(mag_sg) {
      	Out(Get(mag_sg), "\n_space_group_IT_number %s")
      	Out_String("\nloop_\n_space_group_symop_operation_xyz")
      	Out_String("\n_magnetic_space_group_symop_operation_mxmymz")
      	Out_String("\n_magnetic_space_group_symop_operation_timereversal")
      	Out(Get(mag_sp_xyzs_txt),  "%s")
      	} else {
      	Out(Get(sp_grp_char), "\n_space_group_name_H-M_alt %s")
      	Out_String("\nloop_\n_space_group_symop_operation_xyz")
      	Out(Get(sp_xyzs_txt),  "%s")
      	}
			Out_String("\nloop_")
			Out_String("\n_atom_type_symbol")
			Out_String("\n_atom_type_number_in_cell\n")			

			 atom_out file append
			 load out_record out_fmt out_eqn  { 
			" %s" = Get_From_String(Get(current_atom), site);
			" %3.0f\n" = Get_From_String(Get(current_atom), num_posns);
			}
			out file append
			Out_String("\nloop_")
				Out_String("\n_atom_site_label")
      	   Out_String("\n_atom_site_type_symbol")
      	   Out_String("\n_atom_site_fract_x")
      	   Out_String("\n_atom_site_fract_y")
      	   Out_String("\n_atom_site_fract_z")
      	   Out_String("\n_atom_site_occupancy")
      	   Out_String("\n_atom_site_symmetry_multiplicity")      	
      	   Out_String("\n_atom_site_adp_type")
      	   Out_String("\n_atom_site_U_iso_or_equiv\n")      	   
      	   atom_out file append
      	   load out_record out_fmt out_eqn           
      	      {            
      	         "%s" = Get_From_String(Get(current_atom), site);
      	         " %s" = Get_From_String(Get(current_atom), atom);
      	         " %V" = Get_From_String(Get(current_atom), x);
      	         " %V" = Get_From_String(Get(current_atom), y);
      	         " %V" = Get_From_String(Get(current_atom), z);
      	         " %V" = Get_From_String(Get(current_atom), occ);
      	         " %3.0f" = Get_From_String(Get(current_atom), num_posns);
      	         " %s" = "Biso";
      	         " %V\n" = Get_From_String(Get(current_atom), beq);
      	         }
      	   out file append
      	local !density = ((1.6605402 Get(cell_mass)) / Get(cell_volume));
      	Out(density, "\n_exptl_crystal_density_diffrn %2.5f")
      	Out(cell_formula_units_Z, "\n_cell_formula_units_Z %3.0f")
      	local !chemical_formula_weight = Get(cell_mass)/cell_formula_units_Z;:  241.84175
      	Out(chemical_formula_weight, "\n_chemical_formula_weight %11.5f")
      	   if Prm_There(adps) {
      	   Out_String("\nloop_")      	   
      	   Out_String("\n_atom_site_aniso_label")
      	   Out_String("\n_atom_site_aniso_U_11")
      		Out_String("\n_atom_site_aniso_U_22")
      		Out_String("\n_atom_site_aniso_U_33")
      		Out_String("\n_atom_site_aniso_U_12")
      		Out_String("\n_atom_site_aniso_U_13")
      		Out_String("\n_atom_site_aniso_U_23\n")
      		atom_out file append
      	   load out_record out_fmt out_eqn           
      	      {            
      	         "%s" = Get_From_String(Get(current_atom), site);
           			" %V" = Get_From_String(Get(current_atom), u11);
           			" %V" = Get_From_String(Get(current_atom), u22);
           			" %V" = Get_From_String(Get(current_atom), u33);
           			" %V" = Get_From_String(Get(current_atom), u12);
           			" %V" = Get_From_String(Get(current_atom), u13);
           			" %V\n" = Get_From_String(Get(current_atom), u23);
      	         }
      	         }
      if Obj_There(mag_sg) {      		
      out file append      
		Out_String("\n\nloop_")
		Out_String("\n_magnetic_atom_site_label")
		Out_String("\n_magnetic_atom_site_moment_crystalaxis_mx")
      Out_String("\n_magnetic_atom_site_moment_crystalaxis_my")
      Out_String("\n_magnetic_atom_site_moment_crystalaxis_mz")
		atom_out file append
		load out_record out_fmt out_eqn
                 {
		"\n%s" = Get_From_String(Get(current_atom), site);
		"%11.5f" = Get(a) Get_From_String(Get(current_atom), mlx);        
		"%11.5f" = Get(b) Get_From_String(Get(current_atom), mly);        
	         "%11.5f" = Get(c) Get_From_String(Get(current_atom), mlz);        
                  }                  
      } else {
      	         }
      	         
      out file append
      	Out(Get(cif_bonds_angles), "%s")
      	if Obj_There(scrysts) {
      	} else {
      	Out_String("\nloop_ _pd_block_diffractogram_id\n")   
      	for xdds {
      	Out(Get(xdd_path_name), "%s\n") 
      	}}
         Out_String("\n#######End of Structure loop#######")
}
}
macro  Out_CIFdata(file)  {
out file append
			Out_String("\n#######Start of powder data loop#######")
			Out(Get(xdd_path_name), "\ndata_pxrd_%s") 
    		Out(Get(xdd_path_name), "\n_pd_block_id %s")
    		Out_String("\nloop_")
    		Out_String("\n_pd_phase_id")
    		Out_String("\n_pd_phase_block_id")
    		Out_String("\n_pd_phase_mass_%")
    		for strs 
    		{
    			Out(phase_number, "\n%V")
    			Out(phase_number, " phase_%V")
    			Out(Get(sp_grp_char), "_%s")
    			Out(Get(weight_percent), " %1.2f")
    		} 
    		if Obj_There(neutron) {
    			} else {  
    				local !Mu = Get(mixture_MAC) Get(mixture_density_g_on_cm3);
    				Out(Mu, "\n_exptl_absorpt_coefficient_mu %1.2f")    			
    					}		
    		   if And(Obj_There(neutron), Obj_There(pk_xo)) {
    		   } else {
    			Out(Get(lam), "\n_diffrn_radiation_wavelength %1.5f")
    			if Prm_There(LP_Factorval){
    			Out(LP_Factorval, "\n_diffrn_radiation_polarisn_norm %2.3f")
    			}
    			}
       		if Obj_There(neutron) {
    			Out_String("\n_diffrn_radiation_probe neutron")
    			} else {
    			Out_String("\n_diffrn_radiation_probe x-ray")
    			}
    			if Obj_There(scrysts) {  
     		 	Out((Get(gof)^2), "\n_refine_ls_goodness_of_fit_gt %1.5f")  
     		 	' / Max((Sqrt(Yobs)),1)
     		 	xdd_sum !denominator = Sqrt(Yobs);
     		 	xdd_sum !numerator = Abs(Sqrt(Yobs) - Sqrt(Ycalc));
     		 	prm Unweighted_r_factor_based_F = numerator/ denominator;
     		 	Out(Unweighted_r_factor_based_F, "\n_refine_ls_R_factor_gt %1.5f")  
     		 	Out(Get(r_wp), "\n_refine_ls_wR_factor_gt %1.5f")
      		} else {
      		Out((Get(gof)^2), "\n_refine_ls_goodness_of_fit_all %1.5f")
     			Out(Get(r_p), "\n_pd_proc_ls_prof_R_factor %1.5f")
     			Out(Get(r_wp), "\n_pd_proc_ls_prof_wR_factor %1.5f")
     			Out(Get(r_exp), "\n_pd_proc_ls_prof_wR_expected %1.5f") 
				}      		
      		Out_String("\n_pd_calc_method Rietveld")
      		if And(Obj_There(neutron), Obj_There(pk_xo)) {
      			Out(two_theta, "\n_pd_meas_2theta_fixed %1.5f", "(%.9g)")
      			} else { 
   					Out(Get(start_X), "\n_pd_proc_2theta_range_min %11.5f")
   					Out(Get(finish_X), "\n_pd_proc_2theta_range_max %11.5f")
   					Out(Get(x_calculation_step), "\n_pd_proc_2theta_range_inc %11.5f")
   				}
   			Out_String("\nloop_")
   			if And(Obj_There(neutron), Obj_There(pk_xo)) {
					Out_String("\n_pd_proc_point_id")
   				Out_String("\n_pd_meas_time_of_flight")
   				Out_String("\n_pd_proc_d_spacing")
  					Out_String("\n_pd_proc_intensity_total")
  					Out_String("\n_pd_proc_ls_weight")
  					Out_String("\n_pd_calc_intensity_total\n")
  					xdd_out file append 
					load out_record out_fmt out_eqn
					{
	 				  "%1.0f" = Xi;
					  " %11.5f" = X;
	 				  " %11.5f"  = (1.997e-03/((difc/(505.56*Sin(Deg (two_theta/2))))*Sin((Deg (two_theta/2)))))*X;
	 				  " %11.5f" = Yobs;
	 				  "(%1.0f)" = SigmaYobs;
	 				  " %11.5f" = If(Get(weighting) < 1, 1, Get(weighting));
	 				  " %11.5f\n" = Ycalc;	 			 
	 				 }	
	 			} else {	 			
	 				Out_String("\n_pd_proc_point_id")
   			   Out_String("\n_pd_meas_2theta_scan")
  					Out_String("\n_pd_proc_d_spacing")  				
  					Out_String("\n_pd_proc_intensity_total")
  					Out_String("\n_pd_calc_intensity_total")
  					Out_String("\n_pd_proc_ls_weight\n")  					
  						xdd_out file append 
  							load out_record out_fmt out_eqn
  							{
  						 	 "%1.0f" = Xi;
  							 " %11.5f" = X;
  						 	 " %11.5f"   = Lam / (2 Sin((Deg (X/2))));
  						 	 " %11.5f" = Yobs;
  						 	 "(%1.0f)" = SigmaYobs;
  						 	 " %11.5f" = Ycalc;
  						 	 " %11.5f\n" = If(Get(weighting) < 1, 1, Get(weighting));
  						 	 }
  						 	}
  				   out file append
  				   	Out_String("\n#######End of data loop#######")
			Out_String("\nloop_")
      		Out_String("\n_refln_index_h")
      		Out_String("\n_refln_index_k")
      		Out_String("\n_refln_index_l")
      		Out_String("\n_pd_refln_phase_id") 
      		Out_String("\n_refln_d_spacing")
      		Out_String("\n_refln_F_squared_calc")
      		Out_String("\n_refln_F_squared_meas")
      		Out_String("\n_refln_F_squared_sigma")
      		Out_String("\n_refln_include_status\n") 	    	
      		for strs 
      		{ 
      			phase_out file append
      		   load out_record out_fmt out_eqn
      		   {     	   
      		      "%4.0f" = H;
      		      "%4.0f" = K;
      		      "%4.0f" = L;
      		      "%4.0f" = phase_number; 
      		      " %11.5f"  = D_spacing;
      		      " %11.5f" = I_no_scale_pks;
      		      " %11.5f" = Iobs_no_scale_pks;
      		      " %4.5f o\n" = Iobs_no_scale_pks_err;      		        		      
      		   }
      		}      		      	     
      out file append
      Out_String("\n#######End of HKL loop#######")
}
macro Out_FCFSXtal(file)
{
out file append
					Out_String("\n#######Start of FCF loop#######")
					Out(Get(lam), "\n_diffrn_radiation_wavelength %1.5f")
					Out_String("\nloop_")
					Out_String("\n_refln_d_spacing")
					Out_String("\n_refln_index_h")
      			Out_String("\n_refln_index_k")
      			Out_String("\n_refln_index_l") 				 					  					        			
      			Out_String("\n_refln_F_squared_calc")
      			Out_String("\n_refln_F_squared_meas")
      			Out_String("\n_refln_F_squared_sigma")     			
      			Out_String("\n_refln_include_status\n")
	phase_out file append load out_record out_fmt out_eqn
	{
		 "%11.5f"  = D_spacing;
	    "%4.0f" = H;
		 "%4.0f" = K;
		 "%4.0f" = L;
		 ' I_no_scale_pks
		 '	  = Get(scale) Mobs (A01-B11)^2 + (B01+A11)^2; when ignore_differences_in_Friedel_pairs is NOT defined
		 '	  = Get(scale) Mobs (A01^2 + B01^2 + A11^2 + B11^2); when ignore_differences_in_Friedel_pairs IS defined
		 ' If there are no scale_pks then:
		 '   I_no_scale_pks = I_after_scale_pks = Ycalc
		 " %11.4f" = I_no_scale_pks;
		 " %11.4f" = I_after_scale_pks;
		 "%10.2f o" = Iobs_no_scale_pks_err; 
	}
	out file append
	Out_String("\n#######End of FCF loop#######")
}


macro Create_2Th_Ip_FWHM_file(file) {   Create_2Th_Ip_FWHM_file(file, I_no_scale_pks) }
macro Create_2Th_Ip_IB_file(file) {   Create_2Th_Ip_IB_file(file, I_no_scale_pks) }
macro Create_2Th_Ip_FWHM_file(file, wI)
{ 
phase_out file append load out_record out_fmt out_eqn
   {
      " %V"   = Get(xo);
      " %11.5f" = wI;
      " %V\n" = Get(pv_fwhm);
   }   
   }
macro Create_2Th_Ip_IB_file(file, wI)
{
local IBreadth = (Voigt_Integral_Breadth_GL(Get(gauss_fwhm),Get(lor_fwhm)));: 0
   phase_out file append load out_record out_fmt out_eqn
   {
      " %V"   = Get(xo);
      " %11.5f" = wI;
      " %V\n" = IBreadth;
   }
   }

'Bond distance restraints rewritten as restraint equations not penalties

macro Angle_Distance_Restrain_R(& c, & t, t_calc, & tol, & wscale)
   {
      prm = c; : t_calc
      restraint =
         wscale
         IF tol THEN
            IF c < (t - tol) THEN
               (c - (t - tol) )
            ELSE
               IF c > (t + tol) THEN
                  (c - (t + tol))
               ELSE
                  0
               ENDIF
            ENDIF
         ELSE
            (c - t)
         ENDIF
         ;
   }
macro Distance_Restrain_R(sites, c, t, t_calc, tol, wscale)
   {
      #m_ifarg c ""
         #m_unique_not_refine c
      #m_endif
      sites_distance c
         load site_to_restrain { sites }
      Angle_Distance_Restrain_R(c, t, t_calc, tol, wscale)
   }
macro Distance_Restrain_R(sites, t, t_calc, tol, wscale)
   {
      Distance_Restrain_R(sites,, t, t_calc, tol, wscale)
   }
macro Angle_Restrain_R(sites, c, t, t_calc, tol, wscale)
   {
      #m_ifarg c ""
         #m_unique_not_refine c
      #m_endif
      sites_angle c
         load site_to_restrain { sites }
      Angle_Distance_Restrain_R(c, t, t_calc, tol, wscale)
   }
macro Angle_Restrain_R(sites, t, t_calc, tol, wscale)
   {
      Angle_Restrain_R(sites,, t, t_calc, tol, wscale)
   }

macro Out_CIF_STR_rmcprofile(file)
   {
      Out_CIF_STR_rmcprofile(file,)
   }
macro Out_CIF_STR_rmcprofile(file, with_id)
   {
      out file
         Out_String("\ndata_")
         Out(Get(phase_name), "\n_chemical_name_mineral ?%s?")
         Out(Get(a), "\n_cell_length_a  %V")
         Out(Get(b), "\n_cell_length_b  %V")
         Out(Get(c), "\n_cell_length_c  %V")
         Out(Get(al), "\n_cell_angle_alpha %V")
         Out(Get(be), "\n_cell_angle_beta  %V")
         Out(Get(ga), "\n_cell_angle_gamma %V")
         Out(Get(cell_volume), "\n_cell_volume %V")
         Out(Get(sp_grp_char), "\n_symmetry_space_group_name_H-M %s")
         Out_String("\nloop_")
         #m_ifarg with_id ""
            Out_String("\n\t_symmetry_equiv_pos_as_xyz")
            Out(Get(sp_xyzs_txt),  "%-s")
         #m_else
            Out_String("\n\t_symmetry_equiv_pos_site_id\n\t_symmetry_equiv_pos_as_xyz")
            Out(Get(sp_xyzs_txt_with_id),  "%-s")
         #m_endif
         Out_String("\nloop_")
            Out_String("\n_atom_site_label")
'            Out_String("\n_atom_site_type_symbol")
'            Out_String("\n_atom_site_symmetry_multiplicity")
            Out_String("\n_atom_site_fract_x")
            Out_String("\n_atom_site_fract_y")
            Out_String("\n_atom_site_fract_z")
'            Out_String("\n_atom_site_occupancy")
'            Out_String("\n_atom_site_B_iso_or_equiv")
            atom_out file append
               load out_record out_fmt out_eqn
               {
                  "\n%2s" = Get_From_String(Get(current_atom), atom);
 '                 " %s" = Get_From_String(Get(current_atom), atom);
 '                 " %3.0f" = Get_From_String(Get(current_atom), num_posns);
                  " %V" = Get_From_String(Get(current_atom), x);
                  " %V" = Get_From_String(Get(current_atom), y);
                  " %V" = Get_From_String(Get(current_atom), z);
'                  " %V" = Get_From_String(Get(current_atom), occ);
'                  " %V" = Get_From_String(Get(current_atom), beq);
               }
   }


' macro & d_spacing_from_TOF_dayton(& t0, & t1, & t2) {
' ' This macro calculates d spacing from TOF'
' 	(-t1 + Sqrt(t1^2 - 4 t2 (t0 - X))) / (2 t2)
' }

macro Out_d_from_tof_bragg_dayton(filename, tzero_val, difc_val, difa_val)
{
xdd_out filename load out_record out_fmt out_eqn
   {
       " %11.8f  " =  d_spacing_from_TOF_dayton(tzero_val, difc_val, difa_val);
       " %11.8f\n  " = Ycalc;
   } 
}

macro Out_d_Iobs_bragg_dayton(filename, tzero_val, difc_val, difa_val)
{
xdd_out filename load out_record out_fmt out_eqn
   {
       " %11.8f  " =  d_spacing_from_TOF_dayton(tzero_val, difc_val, difa_val);
       " %11.8f\n  " = Yobs;
   } 
}

macro Out_2th_TO_dspacing_Ycalc(filename)
{
xdd_out filename load out_record out_fmt out_eqn
   {
       " %11.6f  " = Lam/(2 * Sin(X Deg/2));
       " %11.6f\n  " = Ycalc;
   } 
}