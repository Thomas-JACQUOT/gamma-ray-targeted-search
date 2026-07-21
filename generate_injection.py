#!/home/thomas-jacquot/.conda/envs/up2dategts/bin/python

import argparse
import subprocess
import os
import datetime
import h5py
from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.finders import ContinuousFtp
import glob
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

from gdt.core.tte import PhotonList, EventList
from gdt.missions.fermi.gbm.response import GbmRsp
from gdt.core.spectra.functions import Band, BrokenPowerLaw
from gdt.core.simulate.profiles import norris, tophat
from gdt.core.simulate.tte import TteSourceSimulator
from gdt.core.data_primitives import ResponseMatrix, Gti
from gdt.core.plot.drm import ResponsePlot


parser = argparse.ArgumentParser()
parser.add_argument("--time", required=True)
parser.add_argument("--format", required=True, choices=['gps', 'fermi', 'datetime'])
parser.add_argument("--inj-ra", required=True)
parser.add_argument("--inj-dec", required=True)
parser.add_argument("--inj-amp", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


if args.format == 'datetime':
    value = datetime.datetime.fromisoformat(args.time)
    print(value)
else:
    value = float(args.time)
trigger = Time(value, format=args.format)

GRB_FERMI_TIME = trigger.fermi

os.makedirs(f"{args.output}", exist_ok=True)
trigger_id = Time(f"{GRB_FERMI_TIME}", format='fermi')
ftp = ContinuousFtp(trigger_id)
ftp.get_cspec(f"{args.output}")
ftp.get_poshist(f"{args.output}")

os.environ["PATH"] += ":/home/shared/gbm-response-generator/bin/"

subprocess.run([
    "SA_GBM_RSP_Gen.pl",
    f"-R{args.inj_ra}",
    f"-D{args.inj_dec}",
    f"-S{GRB_FERMI_TIME}",
    "-V0",
    "-Ccspec",
    f"{args.output}"
], check=True)
start = GRB_FERMI_TIME
stop = start + 20
det_list = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb", "b0", "b1"]
for i in range(len(det_list)):
    rsp_wildcard = f"{args.output}/glg_cspec_*_v00.rsp"
    rsp_files = sorted(glob.glob(rsp_wildcard))
    rsp = GbmRsp.open(rsp_files[i+2]) if i < 12 else GbmRsp.open(rsp_files[-i]) 
    print(rsp)
    drmplot = ResponsePlot(rsp.drm, colorbar=False)
    drmplot.xlim = (8.0, 1000.0)
    drmplot.ylim = (8.0, 1000.0)
    plt.savefig(f"{args.output}/rspplot_{i}.png")
    #breakpoint()
    # (amplitude, Epeak, alpha, beta)
    band_params = (0.001, 300.0, -1.0, -2.8)
    broken_params = (0.001, 300.0, -1.0, -2.8)

    # (amplitude, tstart, trise, tdecay)
    norris_params = (float(args.inj_amp), start, 0.1, 0.5)
    # (amplitude, tstart, tstop)
    tophat_params = (float(args.inj_amp),start, start+1.0)
    src_sim = TteSourceSimulator(rsp, Band(), band_params, norris, norris_params,deadtime=1e-6)
    sim_check = src_sim.simulate(start, stop)
    if sim_check.time_range is None:
        with h5py.File(f"{args.output}/TTE_INJECTION_{i}.hdf5", 'w') as hf:
            hf.create_dataset("times", data=None)
            hf.create_dataset("channels", data=None)
            hf.create_dataset("ebounds", data=None)
            hf.create_dataset("ra", args.inj_ra)
            hf.create_dataset("dec", args.inj_dec)
    else:
        gti = Gti.from_bounds([sim_check.time_range[0]], 
                              [sim_check.time_range[1]])
        src_tte = PhotonList.from_data(sim_check,gti=gti)
        with h5py.File(f"{args.output}/TTE_INJECTION_{i}.hdf5", 'w') as hf:
            hf.create_dataset("times", data=src_tte.data.times)
            hf.create_dataset("channels", data=src_tte.data.channels)
            hf.create_dataset("ebounds", data=src_tte.data.ebounds.as_list())
            hf.create_dataset("ra", args.inj_ra)
            hf.create_dataset("dec", args.inj_dec)
