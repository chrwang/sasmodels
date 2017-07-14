#!/usr/bin/env python
"""
Program to generate sample SANS datasets for neural network training. A
modified version of compare_multi.py.

The program writes datafiles as result_<modelname>_<run_number> to the out/
directory. See example_data.dat for a sample of the file format used.

"""
from __future__ import print_function

import os
import resource
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np  # type: ignore

from . import core
from .compare import (randomize_pars, suppress_pd, make_data,
                      make_engine, get_pars, columnize,
                      constrain_pars, constrain_new_to_old)

MODELS = core.list_models()

# Names for columns of data gotten from simulate
DATA_NAMES = ["Q", "dQ", "I(Q)", "dI(Q)"]


# noinspection PyTypeChecker
def gen_data(name, data, index, N=1, mono=True, cutoff=1e-5,
             base='sasview', output_dir='out_all/'):
    """
    Generates the data for the given model and parameters.

    *name* is the name of the model.

    *data* is the data object giving $q, \Delta q$ calculation points.

    *index* is the active set of points.

    *N* is the number of comparisons to make.

    *cutoff* is the polydispersity weight cutoff to make the calculation
    a little bit faster.

    *base* is the name of the calculation engine to use.
    """

    is_2d = hasattr(data, 'qx_data')
    model_info = core.load_model_info(name)
    pars = get_pars(model_info, use_demo=True)
    header = ('\n"Model","%s","Count","%d","Dimension","%s"'
              % (name, N, "2D" if is_2d else "1D"))
    if not mono:
        header += ',"Cutoff",%g' % (cutoff,)

    if is_2d:
        if not model_info.parameters.has_2d:
            print(',"Only 1-D supported by this model"')
            return

    # A not very clean macro for evaluating the models. They freely use
    # variables from the current scope, even some which have not been defined
    # yet, complete with abuse of mutable lists to allow them to update values
    # in the current scope since nonlocal declarations are not available in
    # python 2.7.
    def exec_model(fn, pars):
        """
        Return the model evaluated at *pars*.  If there is an exception,
        print it and return NaN of the right shape.
        """
        try:
            fn.simulate_data(noise=2, **pars)
            result = np.vstack((fn._data.x, fn._data.dx, fn._data.y,
                                fn._data.dy))
        except Exception:
            traceback.print_exc()
            print("when comparing %s for %d" % (name, seed))
            if hasattr(data, 'qx_data'):
                result = np.NaN * data.data
            else:
                result = np.NaN * data.x
        return result.T

    try:
        calc_base = make_engine(model_info, data, base, cutoff)
    except Exception as exc:
        print('"Error: %s"' % str(exc).replace('"', "'"))
        print('"good","%d of %d","max diff",%g' % (0, N, np.NaN))
        return

    first = True
    q = list()
    iq = list()
    mname = list()
    for k in range(N):
        seed = np.random.randint(int(1e6))
        pars_i = randomize_pars(model_info, pars, seed)
        constrain_pars(model_info, pars_i)
        if 'sasview' in base:
            constrain_new_to_old(model_info, pars_i)
        if mono:
            pars_i = suppress_pd(pars_i)
	count = 0
        while True:
            pars_i.update({'scale': 1, 'background': 1e-5})
            dat = np.vstack(exec_model(calc_base, pars_i))
            if np.in1d(dat, np.NaN).any() or np.in1d(dat, np.Inf).any():
                del dat
                #print("Maths error, retrying")
		count += 1
            else:
                break
	    if count > 2000:
		print("Maths error; bad model: ", name)
		return
        columns = [v for _, v in sorted(pars_i.items())]
        result_dict = OrderedDict()
        result_dict["model"] = name
        result_dict["id"] = k
        columns2 = ['Seed'] + list(sorted(pars_i.keys()))
        params = OrderedDict(zip(columns2, np.insert(columns, 0, seed)))
        result_dict["param"] = params
        data_dict = OrderedDict(
            (n2, col.tolist()) for n2, col in zip(DATA_NAMES, dat.T))
        result_dict["data"] = data_dict

        iq.append(dat.T[2].tolist())
        if first:
            mname.append(name)
            mname.append(N)
            q = dat.T[0].tolist()
            first = False
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
    with open(output_dir + 'result_eval_' + name, 'w') as fd:
        fd.write(str(mname) + '\n')
        fd.write(str(q) + '\n')
        fd.write(str(iq))
    print("Complete")


def print_usage():
    """
    Print the command usage string.
    """
    print("usage: generate_sets.py MODEL COUNT (1dNQ|2dNQ) (CUTOFF|mono) "
          "(single|double|quad)", file=sys.stderr)


def print_models():
    """
    Print the list of available models in columns.
    """
    print(columnize(MODELS, indent="  "))


def print_help():
    """
    Print usage string, the option description and the list of available models.
    """
    print_usage()
    print("""\

MODEL is the model name of the model or one of the model types listed in
sasmodels.core.list_models (all, py, c, double, single, opencl, 1d, 2d,
nonmagnetic, magnetic).  Model types can be combined, such as 2d+single.
$
COUNT is the number of randomly generated parameter sets to try. A value
of "10000" is a reasonable check for monodisperse models, or "100" for
polydisperse models.   For a quick check, use "100" and "5" respectively.

NQ is the number of Q values to calculate.  If it starts with "1d", then
it is a 1-dimensional problem, with log spaced Q points from 1e-3 to 1.0.
If it starts with "2d" then it is a 2-dimensional problem, with linearly
spaced points Q points from -1.0 to 1.0 in each dimension. The usual
values are "1d100" for 1-D and "2d32" for 2-D.

CUTOFF is the cutoff value to use for the polydisperse distribution. Weights
below the cutoff will be ignored.  Use "mono" for monodisperse models.  The
choice of polydisperse parameters, and the number of points in the distribution
is set in compare.py defaults for each model.  Polydispersity is given in the
"demo" attribute of each model.

PRECISION is the floating point precision to use for comparisons. Precision is 
one of fast, single, double for GPU or single!, double!, quad! for DLL.  If no
precision is given, then this defaults to single and double! respectively.

Available models:
""")
    print_models()


def main(argv):
    """
    Main program.
    """
    if len(argv) not in (3, 4, 5):
        print_help()
        return
    #print(len(core.list_models("all")))
    for x in ['adsorbed_layer', 'barbell', 'bcc_paracrystal', 'be_polyelectrolyte', 'binary_hard_sphere', 'broad_peak', 'capped_cylinder', 'core_multi_shell', 'core_shell_bicelle', 'core_shell_bicelle_elliptical', 'core_shell_cylinder', 'core_shell_ellipsoid', 'core_shell_parallelepiped', 'core_shell_sphere', 'correlation_length', 'cylinder', 'dab', 'ellipsoid', 'elliptical_cylinder', 'flexible_cylinder', 'flexible_cylinder_elliptical', 'fractal', 'fractal_core_shell', 'fuzzy_sphere', 'gauss_lorentz_gel', 'gaussian_peak', 'gel_fit', 'guinier', 'guinier_porod', 'hardsphere', 'hayter_msa', 'hollow_cylinder', 'hollow_rectangular_prism', 'hollow_rectangular_prism_thin_walls', 'lamellar', 'lamellar_hg', 'lamellar_hg_stack_caille', 'lamellar_stack_caille', 'lamellar_stack_paracrystal', 'line', 'linear_pearls', 'lorentz', 'mass_fractal', 'mass_surface_fractal', 'mono_gauss_coil', 'multilayer_vesicle', 'onion', 'parallelepiped', 'peak_lorentz', 'pearl_necklace', 'poly_gauss_coil', 'polymer_excl_volume', 'polymer_micelle', 'porod', 'power_law', 'pringle', 'raspberry', 'rectangular_prism', 'rpa', 'sphere', 'spherical_sld', 'spinodal', 'squarewell', 'stacked_disks', 'star_polymer', 'stickyhardsphere', 'surface_fractal', 'teubner_strey', 'triaxial_ellipsoid', 'two_lorentzian', 'two_power_law', 'unified_power_Rg', 'vesicle']:
    #for x in ['fcc_paracrystal', 'flexible_cylinder', 'flexible_cylinder_elliptical', 'fractal', 'fractal_core_shell', 'fuzzy_sphere', 'gauss_lorentz_gel', 'gaussian_peak', 'gel_fit', 'guinier', 'guinier_porod', 'hardsphere', 'hayter_msa', 'hollow_cylinder', 'hollow_rectangular_prism', 'hollow_rectangular_prism_thin_walls', 'lamellar', 'lamellar_hg', 'lamellar_hg_stack_caille', 'lamellar_stack_caille', 'lamellar_stack_paracrystal', 'line', 'linear_pearls', 'lorentz', 'mass_fractal', 'mass_surface_fractal', 'mono_gauss_coil', 'multilayer_vesicle', 'onion', 'parallelepiped', 'peak_lorentz', 'pearl_necklace', 'poly_gauss_coil', 'polymer_excl_volume', 'polymer_micelle', 'porod', 'power_law', 'pringle', 'raspberry', 'rectangular_prism', 'rpa', 'sphere', 'spherical_sld', 'spinodal', 'squarewell', 'stacked_disks', 'star_polymer', 'stickyhardsphere', 'surface_fractal', 'teubner_strey', 'triaxial_ellipsoid', 'two_lorentzian', 'two_power_law', 'unified_power_Rg', 'vesicle']:
        run_model(x, argv)
    #run_model(argv[0], argv)


def run_model(model, argv):
    try:
        model_list = [model] if model in MODELS else core.list_models(model)
    except ValueError:
        print('Bad model %s.  Use model type or one of:' % model,
              file=sys.stderr)
        print_models()
        print('model types: all, py, c, double, single, opencl, 1d, 2d, '
              'nonmagnetic, magnetic')
        return
    try:
        count = int(argv[1])
        is2D = argv[2].startswith('2d')
        assert argv[2][1] == 'd'
        nq = int(argv[2][2:])
        mono = len(argv) <= 3 or argv[3] == 'mono'
        cutoff = float(argv[3]) if not mono else 0
        base = argv[4] if len(argv) > 4 else "single"
    except Exception:
        traceback.print_exc()
        print_usage()
        return
    data, index = make_data({'qmax': 1.0, 'is2d': is2D, 'nq': nq, 'res': 0.03,
                             'accuracy': 'Low', 'view': 'log', 'zero': False})
    for model in model_list:
        gen_data(model, data, index, N=count, mono=mono,
                 cutoff=cutoff, base=base)


if __name__ == "__main__":
    # from .compare import push_seed
    # with push_seed(1): main(sys.argv[1:])
    # core.list_models("1d")
    import logging

    logging.basicConfig(level=logging.INFO)
    time_start = time.clock()
    main(sys.argv[1:])
    time_end = time.clock() - time_start
    print('Total computation time (s): %.2f' % (time_end / 10))
    print('Total memory usage: %.2f' %
          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Units are OS dependent
