import numpy as np
import healpy as hp
import datetime
import argparse
import os
import matplotlib.pyplot as plt

from rich.progress import Progress, TextColumn, TaskProgressColumn, TimeRemainingColumn
from gdt.missions.fermi.gbm.saa import GbmSaa
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from gdt.missions.fermi.gbm.localization import GbmHealPix
from utils import grid_to_healpix
from astropy.coordinates import SkyCoord, get_sun
from gdt.core.plot.sky import EquatorialPlot
from gdt.core.plot.lib import sky_point
from gdt.missions.fermi.time import Time
from search import TargetedSearch
from plots import TargetedLightcurves, Waterfall, plot_orbit
from results import Results
from configuration import InstrumentConfiguration, SearchConfiguration
from skymap import O3_DGAUSS_Model, LigoHealPix



def GetGbmLocalization(search, result, time_ref, include_systematic=True):
    """Compute GBM location with systematic error modeled as
    a double Gaussian (core + tail) shape.

    Args:
        search (TargetedSearch): Search object
        result (Results): Result object
        time_ref (Time): Reference time for the tstart value of result
        include_systematic (bool): Include systematic error when True

    Returns:
        (GbmHealPix)
    """
    # recompute likelihood without sky masking for this timebin
    search.calculate_likelihood(result['tstart'], result['tstart'] + result['duration'], sky_mask=False)

    # compute sky probability for max template
    prob = np.exp(search.like.llr - np.max(search.like.llr))[result['template'], :]

    # project to NSIDE 64 healpix
    proj_prob, _ = grid_to_healpix(
        prob, search.like_points, search.like_frame, nside_out=64)

    # upscale to NSIDE 128
    hires_nside = 128
    hires_npix = hp.nside2npix(hires_nside)
    theta, phi = hp.pix2ang(hires_nside, np.arange(hires_npix))
    upscaled_prob = hp.get_interp_val(proj_prob, theta, phi)

    # build GbmHealpix object
    loc = GbmHealPix.from_data(upscaled_prob, trigtime=time_ref.fermi + result['tstart'],
                               quaternion=search.like_frame.quaternion, scpos=search.like_frame.obsgeoloc)

    # apply systematic error
    if include_systematic:
        systematic = (O3_DGAUSS_Model, result['in_rock'], result['zen'])
        loc = loc.convolve(*systematic)

    # remove Earth region
    loc.remove_earth()

    return loc


parser = argparse.ArgumentParser("plot_targeted_search.py", "Script for plotting the GBM targeted search output")
parser.add_argument('-t', '--time', default=None, help="Time for continuous data search.")
parser.add_argument("--format", required=True, choices=['gps', 'fermi', 'datetime'])
parser.add_argument("--inj-ra", type=float,help="The right ascension of the generated injection in deg")
parser.add_argument("--inj-dec", type=float, help="The declination of the generated injection in deg")
parser.add_argument('-s', '--skymap', default=None, type=str, help="Optional skymap file.")
parser.add_argument('-w', '--search-window-width', default=60, type=float, help="Search window around trigger time in seconds. The search will run from -width/2 until +width/2.")
parser.add_argument('--min-dur', default=0.064, type=float, help="Minimum duration of GRB transient in seconds.")
parser.add_argument('--max-dur', default=8.192, type=float, help="Maximum duration of GRB transient in seconds.")
parser.add_argument('--min-step', default=0.064, type=float, help="Minimum time step size in seconds used to move duration window.")
parser.add_argument('--num-steps', default=8, type=int, help="Sets duration window step size using duration/num_steps for steps larger than --min-step.")
parser.add_argument('-x', '--background-window', default=125.0, type=float, help="NaivePossion background window.")
parser.add_argument('-z', '--background-range', default=[-500, 500], nargs="+", type=float, help="Background fit range(s).")
parser.add_argument('-o', '--results-dir', default='.', type=str, help="Directory for results output.")

args = parser.parse_args()

if args.format == 'datetime':
    value = datetime.datetime.fromisoformat(args.time)
    print(value)
else:
    value = float(args.time)
trigger = Time(value, format=args.format)
progress = Progress(TextColumn("[progress.description]{task.description}"),
                    TaskProgressColumn(), TimeRemainingColumn(elapsed_when_finished=True))


results = Results.open(f"{args.results_dir}/full_results.npz")
filtered_results = Results.open(f"{args.results_dir}/filtered_results.npz")
search_configuration = np.load(f"{args.results_dir}/search_configuration.npz", allow_pickle=True)
search_config = search_configuration['search_config'].item()
instrument_data = search_configuration['instrument_data'].item()
skygrid = search_configuration["skygrid"].item()
search = TargetedSearch(search_config, skygrid)
search.instrument_data['gbm'] = instrument_data 

nai_configs = {det.name: {'channel_edges': [0, 8, 20, 33, 51, 85, 106, 127, 128], 'search_channels': [1, 2, 3, 4, 5, 6]} for det in GbmDetectors.nai()}
bgo_configs = {det.name: {'channel_edges': [0, 8, 21, 40, 65, 90, 112, 124, 128], 'search_channels': [0, 1, 2, 3, 4, 5, 6, 7]} for det in GbmDetectors.bgo()}
gbm_config = InstrumentConfiguration('gbm', nai_configs | bgo_configs)

#search_config = SearchConfiguration(instruments=[gbm_config])
#search_config.settings.update({
#        'win_width': args.search_window_width,
#        'min_loglr': 5,
#        'min_dur': args.min_dur, 'max_dur': args.max_dur,
#        'min_step': args.min_step,'num_steps': args.num_steps,
#        'bkgd_range': args.background_range, 'bkgd_window': args.background_window,
#        'data_range': np.array([-0.5, 0.5]) * (args.search_window_width + args.max_dur)})

poshist = GbmPosHist.open(f"{args.results_dir}/glg_poshist_all_230419_v00.fit")
spacecraft_frames = poshist.get_spacecraft_frame()

print("\nCreating the following plots:")
    
print("\nOrbital plot...")
orbit_filename = os.path.join(args.results_dir, "Orbit.png")
plot_orbit(spacecraft_frames, trigger, orbit_filename, GbmSaa())
print("Done.")

print("\nWaterfall plots...")
w = Waterfall(results, trigger)
loglr_filename = os.path.join(args.results_dir, 'Loglr.png')
w.plot_loglr(loglr_filename, val_min=3.0)
loglr_spec_filename = os.path.join(args.results_dir, 'Loglr_spec.png')
w.plot_loglr(loglr_spec_filename, val_min=3.0, spectra=True)
print("Done.")

print("\nLightcurve plots...")
nai = list(nai_configs.keys())
bgo = list(bgo_configs.keys())
time_range = search_config['search_range']
lcplotter = TargetedLightcurves(instrument_data, trigger)
for i in range(filtered_results.size):
    progress.start()
    task = progress.add_task(f"  Lightcurves for Event {i+1}...", total=12)

    duration, tstart = filtered_results['duration'][i], filtered_results['tstart'][i]

    [(lcplotter.plot_summed(duration, time_range=time_range, event_time=tstart, **kwargs), progress.update(task, advance=1))
        for kwargs in [
        {'filename': os.path.join(args.results_dir, f"Event{i}_Summed_All_NaI_Chan1-6.png"), 'detectors': nai, 'channel_range': (1, 6)},
        {'filename': os.path.join(args.results_dir, f"Event{i}_Summed_Right_NaI_Chan3-4.png"), 'detectors': nai[:6], 'channel_range': (3, 4)},
        {'filename': os.path.join(args.results_dir, f"Event{i}_Summed_Left_NaI_Chan3-4.png"), 'detectors': nai[6:], 'channel_range': (3, 4)},
        {'filename': os.path.join(args.results_dir, f"Event{i}_Summed_All_BGO_Chan0-3.png"), 'detectors': bgo, 'channel_range': (0, 3)}]]

    [(lcplotter.plot_channels(duration, time_range=time_range, event_time=tstart, **kwargs), progress.update(task, advance=1))
        for kwargs in [
        {'filename': os.path.join(args.results_dir, f"Event{i}_Channel_All_NaI_Chan0-7.png"), 'detectors': nai, 'channels': [0, 1, 2, 3, 4, 5, 6, 7]},
        {'filename': os.path.join(args.results_dir, f"Event{i}_Channel_Right_NaI_Chan0-7.png"), 'detectors': nai[:6], 'channels': [0, 1, 2, 3, 4, 5, 6, 7]},
        {'filename': os.path.join(args.results_dir, f"Event{i}_Channel_Left_NaI_Chan0-7.png"), 'detectors': nai[6:], 'channels': [0, 1, 2, 3, 4, 5, 6, 7]},
        {'filename': os.path.join(args.results_dir, f"Event{i}_Channel_All_BGO_Chan0-3.png"), 'detectors': bgo, 'channels': [0, 1, 2, 3]}]]

    [(lcplotter.plot_detectors(duration, time_range=time_range, event_time=tstart, **kwargs), progress.update(task, advance=1))
        for kwargs in [
        {'filename': os.path.join(args.results_dir, f"Event{i}_Detector_All_NaI_Chan1-6.png"), 'detectors': nai, 'channel_range': (1, 6)},
        {'filename': os.path.join(args.results_dir, f"Event{i}_Detector_All_NaI_Chan1-2.png"), 'detectors': nai, 'channel_range': (1, 2)},
        {'filename': os.path.join(args.results_dir, f"Event{i}_Detector_All_NaI_Chan3-4.png"), 'detectors': nai, 'channel_range': (3, 4)},
        {'filename': os.path.join(args.results_dir, f"Event{i}_Detector_All_BGO_Chan1-6.png"), 'detectors': bgo, 'channel_range': (1, 6)}]]

    progress.stop()
    progress.remove_task(task)
print("Done.")

print("\nLocalizations...")
for i, result in enumerate(filtered_results):
    loc = GetGbmLocalization(search, result, trigger)
    loc.write(args.results_dir, filename=f"{args.results_dir}/Event{i+1}_healpix.fit", overwrite=True)

    skyplot = EquatorialPlot()
    skyplot.add_localization(loc, clevels=[0.90, 0.50], gradient=False)
    sky_point(args.inj_ra, args.inj_dec, skyplot.ax, frame="equatorial", marker="*", c="r",label="True sky location")
    plt.savefig(f"{args.results_dir}/Event{i+1}_skymap.png", dpi=300)
    plt.clf()

    # combined localization
    if args.skymap is not None:
        region_prob = loc.region_probability(args.skymap) * 100.0
        print(f"  Event {i+1} Spatial Association: {region_prob:3.1f}%")
        if region_prob > 50.0:
            combined = loc.multiply(loc, args.skymap)
            combined.write(args.results_dir, 
                            filename=f"{args.results_dir}/Event{i+1}_healpix_combined.fit", overwrite=True)

            skyplot = EquatorialPlot()
            skyplot.add_localization(combined, clevels=[0.9, 0.5], gradient=False)
            plt.savefig(f"{args.results_dir}/Event{i+1}_skymap_combined.png", dpi=300)
            plt.clf()
print("Done.")