import argparse
import datetime
import gc
import os
import pickle
import yaml

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
# import matplotlib as mpl; mpl.use('svg')
import seaborn as sns
from matplotlib import colorbar, colors, gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

import data
import read_config
from data import all_fcst_fields, get_dates
from data_generator import DataGenerator as DataGeneratorFull
from evaluation import _init_VAEGAN
from noise import NoiseGenerator
from plots import plot_img_log_coastlines, truncate_colourmap
from rapsd import plot_spectrum1d, rapsd
from setupmodel import setup_model

read_config.set_gpu_mode()  # set up whether to use GPU, and mem alloc mode
downscaling_steps = read_config.read_downscaling_factor()["steps"]

# plotting parameters
value_range_precip = (0.1, 15.0)
value_range_orog = (0.0, 1.0e4)
cmap = ListedColormap(sns.color_palette("YlGnBu", 256))
cmap.set_under('white')
cmap.set_bad('black')
linewidth = 0.4
extent = [19.1, 54.3, -13.7, 24.7]  # left, right, bottom, top
alpha = 0.8
dpi = 200

# colorbar
units = "Rain rate [mm h$^{-1}$]"
cb_tick_loc = np.array([0.1, 0.5, 1, 2, 5, 15])
cb_tick_labels = [0.1, 0.5, 1, 2, 5, 15]

# colormap for LSM -- removes the white end
cmap_lsm = plt.get_cmap('terrain')
cmap_lsm = truncate_colourmap(cmap_lsm, 0, 0.8)

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', type=str,
                    help="directory where model weights are saved")
parser.add_argument('--model_number', type=int,
                    help="model iteration to load", default=313600)
parser.add_argument('--predict_year', type=int,
                    help="year to predict on", default=2019)
parser.add_argument('--num_samples', type=int,
                    help="number of images to generate predictions for", default=1)
parser.add_argument('--pred_ensemble_size', type=int,
                    help="size of prediction ensemble", default=3)
parser.set_defaults(plot_all=False)
parser.add_argument('--plot_all', dest='plot_all', action='store_true',
                    help="Plot all GAN predictions separately, and plot power spectra")
args = parser.parse_args()

log_folder = args.log_folder
model_number = args.model_number
predict_year = args.predict_year
num_samples = args.num_samples
pred_ensemble_size = args.pred_ensemble_size

config_path = os.path.join(log_folder, 'setup_params.yaml')
with open(config_path, 'r') as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
problem_type = setup_params["GENERAL"]["problem_type"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
val_years = setup_params["VAL"]["val_years"]
constant_fields = 2

data_paths = read_config.get_data_paths()

batch_size = 1

weights_fn = os.path.join(log_folder, 'models', f'gen_weights-{model_number:07}.h5')
dates = get_dates(predict_year, start_hour=0, end_hour=168)

assert problem_type == "normal"  # removed autocoarsen from this script

autocoarsen = False
plot_input_title = 'Forecast'
input_channels = 2*len(all_fcst_fields)
assert False  # need to change indices

# load appropriate dataset
data_predict = DataGeneratorFull(dates=dates,
                                 fcst_fields=all_fcst_fields,
                                 start_hour=0,
                                 end_hour=168,
                                 batch_size=batch_size,
                                 log_precip=True,
                                 shuffle=True,
                                 constants=True,
                                 fcst_norm=True)


# initialise model
model = setup_model(mode=mode,
                    arch=arch,
                    downscaling_steps=downscaling_steps,
                    input_channels=input_channels,
                    constant_fields=constant_fields,
                    filters_gen=filters_gen,
                    filters_disc=filters_disc,
                    noise_channels=noise_channels,
                    latent_variables=latent_variables,
                    padding=padding)
gen = model.gen
if mode == "VAEGAN":
    _init_VAEGAN(gen, data_predict, batch_size, latent_variables)
gen.load_weights(weights_fn)

# dataset for benchmarks - not currently used
# data_benchmarks = DataGeneratorFull(dates=dates,
#                                     fcst_fields=data.all_fcst_fields,
#                                     start_hour=0,
#                                     end_hour=168,
#                                     batch_size=batch_size,
#                                     log_precip=False,
#                                     shuffle=True,
#                                     fcst_norm=False)

tpidx_mean = 4*all_fcst_fields.index('tp')  # 4*idx is tp ens mean,
tpidx_stdev = 4*all_fcst_fields.index('tp') + 1  # 4*idx+1 is tp ens stdev

pred = []
seq_real = []
seq_cond = []
seq_const = []
dates_save = []
hours_save = []
data_predict_iter = iter(data_predict)

for ii in range(num_samples):
    inputs, outputs = next(data_predict_iter)

    dates_save.append(data_predict.dates[ii])
    hours_save.append(data_predict.time_idxs[ii])

    # store denormalised inputs, outputs, predictions
    seq_const.append(inputs['hi_res_inputs'])
    input_conditions = inputs['lo_res_inputs'].copy()

    # denormalise precip inputs for plotting
    input_conditions[..., tpidx_mean] = data.denormalise(inputs['lo_res_inputs'][..., tpidx_mean])
    input_conditions[..., tpidx_stdev] = data.denormalise(inputs['lo_res_inputs'][..., tpidx_stdev])

    seq_cond.append(input_conditions)

    truth = outputs['output']
    mask = outputs['mask']
    masked_truth = ma.array(truth, mask=mask)

    # make sure ground truth image has correct dimensions
    sample = np.expand_dims(masked_truth, axis=-1)
    seq_real.append(data.denormalise(sample))

    print(f"sample number {ii+1}")
    print(f"max truth value is {np.max(seq_real[-1])}")

    pred_ensemble = []
    if mode == 'det':  # this is plotting det as a model
        pred_ensemble_size = 1  # can't generate an ensemble with deterministic method
        pred_ensemble.append(data.denormalise(gen.predict(inputs)))  # pretend it's an ensemble so dims match
        pred.append(np.array(pred_ensemble))
    else:
        if mode == 'GAN':
            noise_shape = inputs['lo_res_inputs'][0, ..., 0].shape + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        elif mode == 'VAEGAN':
            noise_shape = inputs['lo_res_inputs'][0, ..., 0].shape + (latent_variables,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        if mode == 'VAEGAN':
            # call encoder once
            mean, logvar = gen.encoder([inputs['lo_res_inputs'], inputs['hi_res_inputs']])
        for jj in range(pred_ensemble_size):
            inputs['noise_input'] = noise_gen()
            if mode == 'GAN':
                gan_inputs = [inputs['lo_res_inputs'], inputs['hi_res_inputs'], inputs['noise_input']]
                pred_ensemble.append(data.denormalise(gen.predict(gan_inputs)))
            elif mode == 'VAEGAN':
                dec_inputs = [mean, logvar, inputs['noise_input'], inputs['hi_res_inputs']]
                pred_ensemble.append(data.denormalise(gen.decoder.predict(dec_inputs)))
            print(f"max predicted value is {np.max(pred_ensemble[-1])}")
        pred.append(np.array(pred_ensemble))

# not used, but could also generate comparison plots of benchmark approaches
# data_benchmarks_iter = iter(data_benchmarks)
# for ii in range(num_samples):
#     inp, outp = next(data_benchmarks_iter)

# plot examples of input fields and predictions
for ii in range(num_samples):
    # len(seq_foo) = num_samples
    # seq[ii].shape = [NHWC], C=input_channels for cond, C=2 for const, C=1 for real, pred
    fcst_total_mean = seq_cond[ii][0, ..., tpidx_mean]
    fcst_total_stdev = seq_cond[ii][0, ..., tpidx_stdev]

    constant_0 = seq_const[ii][0, ..., 0]  # orog
    constant_0 *= 10000.0  # undo orography normalisation
    # constant_1 = seq_const[ii][0, ..., 1]  # lsm

    truth = seq_real[ii][0, :, :, 0]
    pred_0 = pred[ii][0, 0, :, :, 0]  # [sample_images][pred_ensemble_size, N, H, W, C]
    pred_mean = pred[ii][:, 0, :, :, 0].mean(axis=0)  # mean of ensemble members

    # bring precip plots away from 0, since colour scale is logarithmic, and 0 values
    # get confused with NaNs once the log is done
    fcst_total_mean = np.maximum(fcst_total_mean, 1e-6)
    fcst_total_stdev = np.maximum(fcst_total_stdev, 1e-6)
    truth = np.maximum(truth, 1e-6)
    pred_0 = np.maximum(pred_0, 1e-6)
    pred_mean = np.maximum(pred_mean, 1e-6)

    # list of precip-related colorbars
    cbs = []

    plt.figure(figsize=(8, 7), dpi=200)
    # calculate forecast date and valid time, for plot title
    fcst_date = datetime.datetime.strptime(dates_save[ii], "%Y%m%d")
    valid_dt = fcst_date + datetime.timedelta(hours=int(hours_save[ii])*data.HOURS)  # needs to change for 12Z forecasts
    title = f"Forecast {dates_save[ii]}, valid starting {valid_dt.strftime('%Y%m%d %H')}Z"
    plt.suptitle(title, fontsize=16)

    # set up sub-plots
    gs = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax5 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax6 = plt.subplot(gs[1, 2], projection=ccrs.PlateCarree())
    ax = [ax1, ax2, ax3, ax4, ax5, ax6]

    fcst_tmean_ax = ax[0].imshow(fcst_total_mean,
                                 norm=colors.LogNorm(*value_range_precip),
                                 cmap=cmap, origin='lower', extent=extent,
                                 transform=ccrs.PlateCarree(), alpha=alpha)
    ax[0].set_title("Forecast - tp mean")
    ax[0].coastlines(resolution='10m', color='black', linewidth=linewidth)
    cbs.append(plt.colorbar(fcst_tmean_ax, ax=ax[0],
                            norm=colors.LogNorm(*value_range_precip),
                            orientation='horizontal',
                            fraction=0.035, pad=0.04))

    fcst_tstd_ax = ax[1].imshow(fcst_total_stdev,
                                norm=colors.LogNorm(*value_range_precip),
                                cmap=cmap, origin='lower', extent=extent,
                                transform=ccrs.PlateCarree(), alpha=alpha)
    ax[1].set_title('Forecast - tp stdev')
    ax[1].coastlines(resolution='10m', color='black', linewidth=linewidth)
    cbs.append(plt.colorbar(fcst_tstd_ax, ax=ax[1],
                            norm=colors.LogNorm(*value_range_precip),
                            orientation='horizontal',
                            fraction=0.035, pad=0.04))

    OROG = ax[2].imshow(constant_0,
                        cmap="terrain", origin='lower', alpha=alpha)
    ax[2].set_title('Orography')
    foo = plt.colorbar(OROG, ax=ax[2],
                       norm=colors.Normalize(*value_range_orog),
                       orientation='horizontal',
                       fraction=0.04, pad=0.04)
    foo.set_label("Elevation [m]", size=8)

    TRUTH = ax[3].imshow(truth,
                         norm=colors.LogNorm(*value_range_precip),
                         cmap=cmap, origin='lower', extent=extent,
                         transform=ccrs.PlateCarree(), alpha=alpha)
    ax[3].set_title('Ground truth')
    ax[3].coastlines(resolution='10m', color='black', linewidth=linewidth)
    cbs.append(plt.colorbar(TRUTH, ax=ax[3],
                            norm=colors.LogNorm(*value_range_precip),
                            orientation='horizontal',
                            fraction=0.035, pad=0.04))

    PRED = ax[4].imshow(pred_0,
                        norm=colors.LogNorm(*value_range_precip),
                        cmap=cmap, origin='lower', extent=extent,
                        transform=ccrs.PlateCarree(), alpha=alpha)
    ax[4].set_title('GAN - example prediction')
    ax[4].coastlines(resolution='10m', color='black', linewidth=linewidth)
    cbs.append(plt.colorbar(PRED, ax=ax[4],
                            norm=colors.LogNorm(*value_range_precip),
                            orientation='horizontal',
                            fraction=0.035, pad=0.04))

    PRED_mean = ax[5].imshow(pred_mean,
                             norm=colors.LogNorm(*value_range_precip),
                             cmap=cmap, origin='lower', extent=extent,
                             transform=ccrs.PlateCarree(), alpha=0.8)
    ax[5].set_title('GAN - mean prediction')
    ax[5].coastlines(resolution='10m', color='black', linewidth=linewidth)
    cbs.append(plt.colorbar(PRED_mean, ax=ax[5],
                            norm=colors.LogNorm(*value_range_precip),
                            orientation='horizontal',
                            fraction=0.035, pad=0.04))

    for ax_ in ax:
        ax_.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_.tick_params(labelsize=8)

    for cb in cbs:
        cb.set_ticks(cb_tick_loc)
        cb.set_ticklabels(cb_tick_labels)
        cb.set_label(units, size=8)

    # cannot save as pdf - will produce artefacts
    plt.savefig(os.path.join(log_folder,
                             f"prediction-and-inputs-{model_number}-{data_predict.seed}-{ii}.png"),
                bbox_inches='tight')
    plt.close()

if args.plot_all:
    # generate labels for plots
    labels = ["Forecast", "TRUTH"]
    for ii in range(pred_ensemble_size):
        labels.append(f"{mode} pred {ii+1}")

    # plot a range of prediction examples for different downscaling methods
    sequences = []
    for ii in range(num_samples):
        tmp = {}
        tmp['TRUTH'] = np.maximum(seq_real[ii][0, ..., 0], 1e-6)
        tmp["Forecast"] = np.maximum(seq_cond[ii][0, ..., tpidx_mean], 1e-6)
        tmp['dates'] = dates_save[ii]
        tmp['time_idxs'] = hours_save[ii]
        for jj in range(pred_ensemble_size):
            tmp[f"{mode} pred {jj+1}"] = np.maximum(pred[ii][jj][0, ..., 0], 1e-6)
        sequences.append(tmp)

    fname = "sequences-" + str(model_number) + "-" + str(num_samples) + ".pickle"
    fnamefull = os.path.join(log_folder, fname)

    with open(fnamefull, 'wb') as f:
        pickle.dump(sequences, f)

    num_cols = num_samples
    num_rows = len(labels)
    spacing = 10
    plt.figure(figsize=(1.5*num_cols, 1.5*num_rows), dpi=300)

    gs = gridspec.GridSpec(spacing*num_rows+1, spacing*num_cols,
                           wspace=0.5, hspace=0.5)

    for kk in range(num_samples):
        for ii in range(len(labels)):
            plt.subplot(gs[(spacing*ii):(spacing+spacing*ii),
                           (spacing*kk):(spacing+spacing*kk)],
                        projection=ccrs.PlateCarree())
            ax = plt.gca()
            ax.coastlines(resolution='10m', color='black', linewidth=linewidth)
            plot_img_log_coastlines(sequences[kk][labels[ii]],
                                    value_range_precip=value_range_precip,
                                    cmap=cmap,
                                    extent=extent,
                                    alpha=alpha)
            if ii == 0:
                title = dates_save[kk][:4] + '-' + dates_save[kk][4:6] + '-' + dates_save[kk][6:8] + ' ' + str(hours_save[kk]) + ' time period'
                plt.title(title, fontsize=9)

            if kk == 0:
                ax.set_ylabel(labels[ii])  # cartopy takes over the xlabel and ylabel
                ax.set_yticks([])  # this weird hack restores them. WHY?!?!

    plt.suptitle('Example predictions for different input conditions')

    cax = plt.subplot(gs[-1, 1:-1]).axes
    cb = colorbar.ColorbarBase(cax, norm=colors.LogNorm(*value_range_precip), cmap=cmap, orientation='horizontal')
    cb.set_ticks(cb_tick_loc)
    cb.set_ticklabels(cb_tick_labels)
    cax.tick_params(labelsize=12)
    cb.set_label(units, size=12)

    # cannot save as pdf - will produce artefacts
    plt.savefig(os.path.join(log_folder,
                             f"predictions-{model_number}-{data_predict.seed}.png"),
                bbox_inches='tight')
    plt.close()
    gc.collect()

# power spectrum plots
if args.plot_all:
    # colours = ['plum', 'palevioletred', 'lightslategrey', 'coral', 'lightblue', 'darkseagreen', 'mediumturquoise', 'purple', 'navy']
    plt.style.use('seaborn-colorblind')
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_scales = [512, 256, 128, 64, 32, 16, 8, 4]
    # create a PdfPages object to save multiple plots to same pdf
    pdf = PdfPages(os.path.join(log_folder,
                                f"RAPSD-{model_number}-{data_predict.seed}.pdf"))

    for kk in range(num_samples):
        fig, ax = plt.subplots()
        # this iterates over the minimum of len(labels) [pred_ensemble_size + 2]
        # and colours [6], i.e., it will show at most 4 GAN predictions
        for ii, color in zip(range(len(labels)), colours):
            R_1, freq_1 = rapsd(sequences[kk][labels[ii]], fft_method=np.fft, return_freq=True)
            # Plot the observed power spectrum and the model
            plot_spectrum1d(freq_1,
                            R_1,
                            x_units="km",
                            y_units="dBR",
                            color=color,
                            ax=ax,
                            label=labels[ii],
                            wavelength_ticks=plot_scales)

            plt.legend()

        ax.set_title(f"Radially averaged log-power spectrum - {kk+1}")
        # save the current figure
        pdf.savefig(fig)
    plt.close()
    pdf.close()
