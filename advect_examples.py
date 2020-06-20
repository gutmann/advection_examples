#!/usr/bin/env python

"""
SYNOPSIS

    advect_example.py [-h] [--verbose] [-v, --version]

DESCRIPTION

    Run a specified advection algorithm and plot the results.

EXAMPLES

    advect_example.py -a upwind --step -n 100 -t 1000 -u 0.25 --plot

AUTHOR

    Ethan Gutmann - gutmann@ucar.edu

LICENSE

    This script is in the public domain.

VERSION


"""
from __future__ import absolute_import, print_function, division

import sys
import os
import traceback
import argparse

import numpy as np
import matplotlib.pyplot as plt

from advect_core import upwind, adamsbashforth, rungakutta

global verbose
verbose=False

def initialize(function, nx):
    if function == "sine":
        q = np.sin(np.arange(nx) / (nx-1) * 2 * np.pi)

    elif function == "step":
        q = np.zeros(nx)
        q[:int(nx/2)] = 1

    else:
        print(f"Unknown initialization function: {function}")
        print("-------------------------------")
        raise ValueError

    q[0] = q[-1]
    return q


def main (algorithm, function, nx, nt, u, plot):

    if verbose: print("Initialization")
    q = initialize(function, nx+1)

    if algorithm == "adamsbashforth":
        qold2 = q.copy()
        qold1 = q.copy()
        qsave = q.copy()

    if verbose: print("Running")

    for i in range(nt):

        if algorithm == "upwind":
            q[:] = upwind(q, u)

        elif algorithm == "rungakutta":
            q[:] = rungakutta(q, u)

        elif algorithm == "adamsbashforth":
            qsave[:] = q
            q[:] = adamsbashforth(q, u, qold1, qold2)
            qold2[:] = qold1
            qold1[:] = qsave

        else:
            print(f"Unknown algorithm: {algorithm}")
            print("-------------------------------")
            raise ValueError



    if plot:
        if verbose: print("Plotting")
        init = initialize(function, nx+1)
        plt.plot(init[1:], color="black", label="Init.")
        plt.plot(q[1:], label=algorithm, color="C1")
        plt.legend()
        imean = init[1:].mean()
        qmean = q[1:].mean()
        plt.plot([0,nx],[imean, imean], ":", color="black")
        plt.plot([0,nx],[qmean, qmean], ":", color="C1")
        plt.title(f"nt={nt}, u={u}")
        plt.savefig(f"{algorithm}_{function}_{nx}_{nt}_{u}.png")


if __name__ == '__main__':
    try:
        parser= argparse.ArgumentParser(description='Run an advection algorithm. ',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-a', dest='algorithm', nargs="?", action='store', default="upwind",
                            help="name of advection algorithm [upwind, rungakutta, adamsbashforth]")
        parser.add_argument('-f', dest='function', nargs="?", action='store', default="step",
                            help="name of initial shape function [step, sine]")
        parser.add_argument('-n', dest="nx", nargs="?", action='store', default=100, type=int, help="number of x grid points")
        parser.add_argument('-t', dest="nt", nargs="?", action='store', default=800, type=int, help="number of time steps")
        parser.add_argument('-c', dest="nc", nargs="?", action='store', default=None, type=int, help="number of times to cycle through the domain")
        parser.add_argument('-u', dest="u", nargs="?", action='store', default=0.25, type=float, help="wind speed (gridcells / timestep)")
        parser.add_argument ('--plot', action='store_true',
                default=False, help='plot output', dest='plot')

        parser.add_argument('-v', '--version',action='version',
                version='advect_examples 1.0')
        parser.add_argument ('--verbose', action='store_true',
                default=False, help='verbose output', dest='verbose')
        args = parser.parse_args()

        verbose = args.verbose

        if args.nc is not None:
            args.nt = int(args.nx / args.u * args.nc)
            if verbose: print(f"Updating to use {args.nt} timesteps")


        exit_code = main(args.algorithm, args.function, args.nx, args.nt, args.u, args.plot)

        if exit_code is None:
            exit_code = 0
        sys.exit(exit_code)
    except KeyboardInterrupt as e: # Ctrl-C
        raise e
    except SystemExit as e: # sys.exit()
        raise e
    except Exception as e:
        print('ERROR, UNEXPECTED EXCEPTION')
        print(str(e))
        traceback.print_exc()
        os._exit(1)
