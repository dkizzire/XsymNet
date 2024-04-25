# We'll need numpy and matplotlib for plotting our results
import numpy as np
import matplotlib.pyplot as plt

# A least squares fitting algorithm from scipy
from scipy.optimize.minpack import leastsq

# DiffPy-CMI modules for building a fitting recipe
from diffpy.structure import loadStructure
from diffpy.srfit.pdf import PDFContribution
from diffpy.srfit.fitbase import FitRecipe, FitResults
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.scatteringfactortable import SFTNeutron


def run_pdf_calc(structure_fn, stype, qmax, qmin, qdamp, qbroad, rmax, rmin, uiso):
    '''
    Parameters
    ----------
    structure_fn: str
        input structure cif filename.
    output_stem_fn: str
        output stem filename.
    stype : str
        Neutron or xray scattering type, input 'N' or 'X'.
    qmax : float
        Fourier transform Qmax, in units of A^-1.
    qmin : float
        Fourier transform Qmin, in units of A^-1.
    qdamp :float
        Instrument resolution parameter Qdamp, in units of A^-1.
    qbroad :float
        Instrument resolution parameter Qbroad, in units of A^-1.
    rmax : float
        r-range max.
    rmin : float
        r-range min.
    uiso : float
        Uiso ADP value.

    Returns
    -------
    result_data : str
        structure refinement result.
    '''
    print("INFO: Start PDF Auto Simulation.")
    print("structure filename: " + structure_fn)
    print("******************************")

    diffpy_structure = loadStructure(structure_fn)
    diffpy_structure.element = diffpy_structure.element.rstrip('+-0123456789.')

    diffpy_structure.Uisoequiv = uiso
    cfg = {
            'qmax':qmax,
            'qmin':qmin,
            'rmin':rmin,
            'rmax':rmax,
           'qdamp':qdamp,
           'qbroad':qbroad,
    }

    pc = PDFCalculator(**cfg)
    if stype == 'N':
        ntb = SFTNeutron()
        pc.scatteringfactortable = ntb
    r, gcalc = pc(diffpy_structure)
    # plot_pdf_calc(r, gcalc, output_stem_fn)
    #
    # result_fn = output_stem_fn + '.dat'
    # with open(result_fn, "w+") as output_pdf_result:
    #     np.savetxt(output_pdf_result, np.array([r,gcalc]).T, fmt=['%f','%f'], header = "r   G")
    return r, gcalc



def run_nxpdf_calc(structure_fn, stype, qmax, qmin, qdamp, qbroad, rmax, rmin, uiso):
    '''
    Parameters
    ----------
    structure_fn: str
        input structure cif filename.
    output_stem_fn: str
        output stem filename.
    stype : str
        Neutron or xray scattering type, input 'N' or 'X'.
    qmax : float
        Fourier transform Qmax, in units of A^-1.
    qmin : float
        Fourier transform Qmin, in units of A^-1.
    qdamp :float
        Instrument resolution parameter Qdamp, in units of A^-1.
    qbroad :float
        Instrument resolution parameter Qbroad, in units of A^-1.
    rmax : float
        r-range max.
    rmin : float
        r-range min.
    uiso : float
        Uiso ADP value.

    Returns
    -------
    result_data : str
        structure refinement result.
    '''
    print("INFO: Start PDF Auto Simulation.")
    print("structure filename: " + structure_fn)
    print("******************************")

    diffpy_structure = loadStructure(structure_fn)
    diffpy_structure.element = diffpy_structure.element.rstrip('+-0123456789.')

    diffpy_structure.Uisoequiv = uiso
    cfg = {
            'qmax':qmax,
            'qmin':qmin,
            'rmin':rmin,
            'rmax':rmax,
           'qdamp':qdamp,
           'qbroad':qbroad,
    }

    pcx = PDFCalculator(**cfg)
    pcn = PDFCalculator(**cfg)
    ntb = SFTNeutron()
    pcn.scatteringfactortable = ntb

    r, gcalcx = pcx(diffpy_structure)
    r, gcalcn = pcn(diffpy_structure)
    # plot_pdf_calc(r, gcalc, output_stem_fn)
    #
    # result_fn = output_stem_fn + '.dat'
    # with open(result_fn, "w+") as output_pdf_result:
    #     np.savetxt(output_pdf_result, np.array([r,gcalc]).T, fmt=['%f','%f'], header = "r   G")
    return r, gcalcx, gcalcn
