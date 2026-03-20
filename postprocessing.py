import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import h5py
import datetime

from gdt.core.plot.lightcurve import Lightcurve
from gdt.missions.fermi.time import Time
from gdt.core.tte import PhotonList, EventList
from gdt.core.data_primitives import ResponseMatrix, Gti, Ebounds
from gdt.core.binning.unbinned import bin_by_time


def define_rows_cols_subplot(nplots):
    cols = int(np.ceil(np.sqrt(nplots)))
    rows = int(np.ceil(nplots/cols))
    return cols, rows

def plot_TTE_heat_maps(times, channels, bins, det_list, dir, title, filename, mask=None, zoom = False):
    cols, rows = define_rows_cols_subplot(len(times))
    print("rows : ", rows)
    print("cols : ", cols)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(20,12))
    if rows*cols != len(times):
        for i in range(rows*cols-len(times)):
            axs[-1,-(i+1)].axis("Off")
    if mask is not None:
        det_list = det_list[mask]

    if zoom: 
        for i in tqdm(range(rows)):
            for j in tqdm(range(cols)):
                if cols*i+j < len(times):
                    hb = axs[i,j].hist2d(
                        times[cols*i+j][(times[cols*i+j] >= -30) & (times[cols*i+j] <= 30)], 
                        channels[cols*i+j][(times[cols*i+j] >= -30) & (times[cols*i+j] <= 30)],
                        bins=bins, cmap="inferno")
                    fig.colorbar(hb[3], ax=axs[i,j])
                    axs[i,j].set_title(f"{det_list[cols*i+j]}")
                else:
                    pass
        fig.suptitle(title, fontsize=25)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.savefig(f"{dir}/{filename}")

    else:
        for i in tqdm(range(rows)):
            for j in tqdm(range(cols)):
                if cols*i+j < len(times):
                    hb = axs[i,j].hist2d(times[cols*i+j], channels[cols*i+j],bins=bins, cmap="inferno")
                    fig.colorbar(hb[3], ax=axs[i,j])
                    axs[i,j].set_title(f"{det_list[cols*i+j]}")
                else:
                    pass
        fig.suptitle(title, fontsize=25)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.savefig(f"{dir}/{filename}")

def plot_TTE_ligthcurve(times, channels, ebounds, det_list, dir, title, filename, trigtime, mask = None):
    events = [EventList(
        times=times[i][(times[i] > -30) & (times[i] < 30)], 
        channels=channels[i][(times[i] > -30) & (times[i] < 30)], 
        ebounds=ebounds[i])
        for i in range(len(times))
    ]
    print(events)
    photon_list = [PhotonList.from_data(
                event,
                gti=Gti.from_bounds([event.time_range[0]], [event.time_range[1]])
            ) for event in events
    ]
    print(photon_list)
    cols, rows = define_rows_cols_subplot(len(times))
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(20,12))
    if rows*cols != len(times):
        for i in range(rows*cols-len(times)):
            axs[-1,-(i+1)].axis("Off")
    if mask is not None:
        det_list = det_list[mask]
    for i in tqdm(range(rows)):
        for j in tqdm(range(cols)):
            if cols*i+j < len(times):
                pha_tte_sim = photon_list[cols*i+j].to_phaii(bin_by_time, 0.064)
                print(pha_tte_sim)
                axs[i,j] = Lightcurve(data=pha_tte_sim.to_lightcurve(), ax=axs[i,j])
                axs[i,j].ax.set_title(f"{det_list[cols*i+j]}")
            else:
                pass
    fig.suptitle(title, fontsize=25)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig(f"{dir}/{filename}")

parser = argparse.ArgumentParser()
parser.add_argument("--file-path", required=True)
parser.add_argument("--inj-ra", required=True)
parser.add_argument("--inj-dec", required=True)
parser.add_argument("--time", required=True)
parser.add_argument("--format", required=True, choices=[None, 'gps', 'fermi', 'datetime'])

args = parser.parse_args()

if args.format == 'datetime':
    value = datetime.datetime.fromisoformat(args.time)
    print(value)
else:
    value = float(args.time)
trigger = Time(value, format=args.format)

det_list = np.array(["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb", "b0", "b1"])
sim_channels = [None] * 14
sim_times = [None] * 14
sim_ebounds = [None] * 14
tte_sim_channels = [None] * 14
tte_channels = [None] * 14
tte_sim_ebounds = [None] * 14
tte_sim_times = [None] * 14
tte_times = [None] * 14
tte_ebounds = [None] * 14

with h5py.File(f"{args.file_path}/sim_data.hdf5","r") as hf:
    sim_mask = np.array([hf['flag'][:]])
    sim_mask = sim_mask.astype(bool)
    for i in range(len(det_list[sim_mask[0]])):
        sim_times[i] =  hf[f'{det_list[i]}/times'][:]
        sim_channels[i] =  hf[f'{det_list[i]}/channels'][:]
        sim_ebounds[i] =  Ebounds.from_list(hf[f'{det_list[i]}/ebounds'][:])
    
with h5py.File(f"{args.file_path}/tte_sim_data.hdf5","r") as hf:
    for i in range(len(det_list)):
        tte_sim_times[i] = hf[f'{det_list[i]}/times'][:]
        tte_sim_channels[i] = hf[f'{det_list[i]}/channels'][:]
        tte_sim_ebounds[i] = Ebounds.from_list(hf[f'{det_list[i]}/ebounds'][:])

with h5py.File(f"{args.file_path}/tte_data.hdf5","r") as hf:
    for i in range(len(det_list)):
        tte_times[i] = hf[f'{det_list[i]}/times'][:]
        tte_channels[i] = hf[f'{det_list[i]}/channels'][:]
        tte_ebounds[i] = Ebounds.from_list(hf[f'{det_list[i]}/ebounds'][:])

bins = 128
plot_TTE_heat_maps(times=sim_times, channels=sim_channels, bins=bins, 
                    det_list=det_list, dir=args.file_path,
                    title="Energy channels VS times Simulated TTE for each detector",
                    filename="Energy_chan_VS_times_tte_sim_chan",
                    mask=sim_mask[0]
)

plot_TTE_heat_maps(times=tte_times, channels=tte_channels, bins=bins,
                    det_list=det_list, dir=args.file_path,
                    title="Energy channels VS times TTE data for each detector",
                    filename="Energy_chan_VS_times_tte_data_chan",
                    zoom=True
)

plot_TTE_heat_maps(times=tte_sim_times, channels=tte_sim_channels, bins=bins, 
                    det_list=det_list, dir=args.file_path,
                    title="Energy channels VS times TTE data + Simulated TTE for each detector",
                    filename="Energy_chan_VS_times_tte_sim_data_chan",
                    zoom=True
)
print(sim_mask, sim_mask[0])
print("Plotting Simulated Lightcurves...")
plot_TTE_ligthcurve(times=sim_times, channels=sim_channels,
                    ebounds=sim_ebounds, det_list=det_list, dir=args.file_path,
                    title="Injection lightcurve plots in each detector",
                    filename="Injection lightcurve plots in each detector.png",
                    mask=sim_mask[0], trigtime=trigger.fermi
)

print("Plotting Data + Simulated Lightcurves...")
plot_TTE_ligthcurve(times=tte_sim_times, channels=tte_sim_channels, ebounds=tte_sim_ebounds,
                    det_list=det_list,
                    dir=args.file_path,
                    title="Data + Injection lightcurve plots in each detector",
                    filename="Data + Injection lightcurve plots in each detector.png",
                    trigtime=trigger.fermi
)

print("Plotting Data Lightcurves...")
plot_TTE_ligthcurve(times=tte_times, channels=tte_channels, ebounds=tte_ebounds,
                    det_list=det_list,
                    dir=args.file_path,
                    title="Data lightcurve plots in each detector",
                    filename="Data lightcurve plots in each detector.png",
                    trigtime=trigger.fermi
)
print('Done.')