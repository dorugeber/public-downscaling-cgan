""" File for handling data loading and saving. """
import os
import datetime
import pickle

import numpy as np
import netCDF4 as nc
import xarray as xr

import read_config


data_paths = read_config.get_data_paths()
TRUTH_PATH = data_paths["GENERAL"]["TRUTH_PATH"]
FCST_PATH = data_paths["GENERAL"]["FORECAST_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]

all_fcst_fields = ['cape', 'cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp', 'u700', 'v700']
accumulated_fields = ['cp', 'ssr', 'tp']
nonnegative_fields = ['cape', 'cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp']

HOURS = 6  # 6-hr data


# utility function; generator to iterate over a range of dates
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def denormalise(x):
    """
    Undo log-transform of rainfall.  Also cap at 100 (feel free to adjust according to application!)
    """
    return np.minimum(10**x - 1.0, 100.0)


def logprec(y, log_precip=False):
    if log_precip:
        return np.log10(1.0+y)
    else:
        return y


def get_dates(year,
              start_hour,
              end_hour):
    '''
    Returns list of valid forecast start dates for which 'truth' data
    exists, given the other input parameters. If truth data is not available
    for certain days/hours, this will not be the full year. Dates are returned
    as a list of YYYYMMDD strings.

    Parameters:
        year (int): forecasts starting in this year
        start_hour (int): Lead time of first forecast desired
        end_hour (int): Lead time of last forecast desired
        seq_length (int): Length of forecast sequence, in frames
    '''
    # sanity checks for our dataset
    assert year in (2018, 2019, 2020, 2021)
    assert start_hour >= 0
    assert end_hour <= 168
    assert start_hour % HOURS == 0
    assert end_hour % HOURS == 0
    assert end_hour > start_hour

    # Build "cache" of truth data dates/times that exist
    truth_cache = set()
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year+1, 1, end_hour//24 + 2)  # go a bit into following year
    for curdate in daterange(start_date, end_date):
        datestr = curdate.strftime('%Y%m%d')
        for hr in range(0, 24, HOURS):
            fname = f"{datestr}_{hr:02}"
            if os.path.exists(os.path.join(TRUTH_PATH, f"{fname}.nc4")):
                truth_cache.add(fname)

    # Now work out which IFS start dates to use. For each candidate start date,
    # work out which truth dates+times are needed, and check if they exist.
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year+1, 1, 1)
    valid_dates = []

    for curdate in daterange(start_date, end_date):
        # Check interval by interval.  Not particularly efficient, but almost
        # certainly not a bottleneck, since we're just repeatedly testing
        # inclusion in a Python set, and never hitting the disk
        valid = True

        for hr in range(start_hour, end_hour, HOURS):
            # implicitly assumes 00Z forecasts; needs editing for 12Z
            truth_dt = curdate + datetime.timedelta(hours=hr)
            # this works for our specific naming convention, where e.g.,
            # 20190204_06 contains the truth data for hours 06-12 on date 20190204
            if truth_dt.strftime('%Y%m%d_%H') not in truth_cache:
                valid = False
                break
        if valid:
            datestr = curdate.strftime('%Y%m%d')
            valid_dates.append(datestr)

    return valid_dates


def load_truth_and_mask(date,
                        time_idx,
                        seq_length,
                        log_precip=False):
    '''
    Returns a single (truth, mask) sequence of data.
    Parameters:
        date: forecast start date
        time_idx: forecast 'valid time' array index of first frame
        seq_length: length of sequence, in frames
        log_precip: whether to apply log10(1+x) transformation
    '''
    # convert date and time_idx to get the correct truth files
    fcst_date = datetime.datetime.strptime(date, "%Y%m%d")
    ys = []
    masks = []

    for ii in range(seq_length):
        valid_dt = fcst_date + datetime.timedelta(hours=int(time_idx+ii)*HOURS)  # needs to change for 12Z forecasts
        fname = valid_dt.strftime('%Y%m%d_%H')
        data_path = os.path.join(TRUTH_PATH, f"{fname}.nc4")

        ds = xr.open_dataset(data_path)
        da = ds["precipitationCal"]
        y = da.values  # shape H x W
        ys.append(y)  # don't just do ys.append(da.values), since we want shape
        ds.close()  # to create the mask, a few lines below

        # mask: False for valid truth data, True for invalid truth data
        # (compatible with the NumPy masked array functionality)
        # if all data is valid:
        mask = np.full(y.shape, False, dtype=bool)
        masks.append(mask)

    y = np.array(ys)  # shape T x H x W
    mask = np.array(masks)

    if log_precip:
        return np.log10(1+y), mask
    else:
        return y, mask


def load_hires_constants():
    oro_path = os.path.join(CONSTANTS_PATH, "elev.nc")
    df = xr.load_dataset(oro_path)
    # Orography in m.  Divide by 10,000 to give O(1) normalisation
    z = df["elevation"].values
    z /= 10000.0
    df.close()

    lsm_path = os.path.join(CONSTANTS_PATH, "lsm.nc")
    df = xr.load_dataset(lsm_path)
    # LSM is already 0:1
    lsm = df["lsm"].values
    df.close()

    return np.stack([z, lsm], axis=-1)  # shape H x W x 2


def load_fcst_truth_batch(dates_batch,
                          time_idx_batch,
                          seq_length,
                          fcst_fields=all_fcst_fields,
                          log_precip=False,
                          norm=False):
    '''
    Returns a batch of (forecast, truth, mask) data, although usually the batch size is 1
    Parameters:
        dates_batch (iterable of strings): Dates of forecasts
        time_idx_batch (iterable of ints): Corresponding 'valid_time' array indices of first frame
        seq_length (int): Length of forecast/truth series, in frames
        fcst_fields (list of strings): The fields to be used
        log_precip (bool): Whether to apply log10(1+x) transform to precip-related forecast fields, and truth
        norm (bool): Whether to apply normalisation to forecast fields to make O(1)
    '''
    batch_x = []  # forecast
    batch_y = []  # truth
    batch_mask = []  # mask

    for time_idx, date in zip(time_idx_batch, dates_batch):
        batch_x.append(load_fcst_stack(fcst_fields, date, time_idx, seq_length, log_precip=log_precip, norm=norm))
        truth, mask = load_truth_and_mask(date, time_idx, seq_length, log_precip=log_precip)
        batch_y.append(truth)
        batch_mask.append(mask)

    return np.array(batch_x), np.array(batch_y), np.array(batch_mask)


def load_fcst(field,
              date,
              time_idx,
              seq_length,
              log_precip=False,
              norm=False):
    '''
    Returns forecast field data for the given date and time intervals.

    Four channels are returned for each field:
        - instantaneous fields: mean and standard deviation of the field, at the start and end of the interval
        - accumulated field: mean and standard deviation of increment over the interval, and the last two channels are all 0
    '''

    yearstr = date[:4]
    year = int(yearstr)
    ds_path = os.path.join(FCST_PATH, yearstr, f"{field}.nc")

    # open using netCDF
    nc_file = nc.Dataset(ds_path, mode="r")
    all_data_mean = nc_file[f"{field}_mean"]
    all_data_sd = nc_file[f"{field}_sd"]
    # data is stored as [day of year, valid time index, lat, lon]

    # calculate first index (i.e., day of year, with Jan 1 = 0)
    fcst_date = datetime.datetime.strptime(date, "%Y%m%d").date()
    fcst_idx = fcst_date.toordinal() - datetime.date(year, 1, 1).toordinal()

    if field in accumulated_fields:
        # return (mean, sd, 0, 0) for each time.  zero fields are so that each field returns 4 channels (T x H x W x 4)
        # accumulated fields have been pre-processed s.t. data[:, j, :, :] has accumulation between times j and j+1
        data1 = all_data_mean[fcst_idx, time_idx:time_idx+seq_length, :, :]
        data2 = all_data_sd[fcst_idx, time_idx:time_idx+seq_length, :, :]
        data3 = np.zeros(data1.shape)
        data = np.stack([data1, data2, data3, data3], axis=-1)
    else:
        # return mean_start, sd_start, mean_end, sd_end, for each time (T x H x W x 4)
        temp_data_mean = all_data_mean[fcst_idx, time_idx:time_idx+seq_length+1, :, :]
        temp_data_sd = all_data_sd[fcst_idx, time_idx:time_idx+seq_length+1, :, :]
        data1 = temp_data_mean[:-1, :, :]
        data2 = temp_data_sd[:-1, :, :]
        data3 = temp_data_mean[1:, :, :]
        data4 = temp_data_sd[1:, :, :]
        data = np.stack([data1, data2, data3, data4], axis=-1)

    nc_file.close()

    if field in nonnegative_fields:
        data = np.maximum(data, 0.0)  # eliminate any data weirdness/regridding issues

    if field in ["tp", "cp"]:
        # precip is measured in metres, so multiply to get mm
        data *= 1000
        data /= HOURS  # convert to mm/hr
    elif field in accumulated_fields:
        # for all other accumulated fields [just ssr for us]
        data /= (HOURS*3600)  # convert from a 6-hr difference to a per-second rate

    if field in ["tp", "cp"] and log_precip:
        return logprec(data, log_precip)
    elif norm:
        # apply transformation to make fields O(1), based on historical
        # forecast data from one of the training years
        if fcst_norm is None:
            raise RuntimeError("Forecast normalisation dictionary has not been loaded")
        if field in ["mcc"]:
            # already 0-1
            return data
        elif field in ["sp", "t2m"]:
            # these are bounded well away from zero, so subtract mean from ens mean (but NOT from ens sd!)
            data[..., 0] -= fcst_norm[field]["mean"]
            data[..., 2] -= fcst_norm[field]["mean"]
            return data/fcst_norm[field]["std"]
        elif field in nonnegative_fields:
            return data/fcst_norm[field]["max"]
        else:
            # winds
            return data/max(-fcst_norm[field]["min"], fcst_norm[field]["max"])
    else:
        return data


def load_fcst_stack(fields,
                    date,
                    time_idx,
                    seq_length,
                    log_precip=False,
                    norm=False):
    '''
    Returns forecast fields, for the given date and time intervals.
    Each field returned by load_fcst has four channels (see load_fcst for details),
    then these are concatentated to form an array of T x H x W x 4*len(fields)
    '''
    field_arrays = []
    for f in fields:
        field_arrays.append(load_fcst(f, date, time_idx, seq_length, log_precip=log_precip, norm=norm))
    return np.concatenate(field_arrays, axis=-1)


def get_fcst_stats_slow(field, year=2018):
    '''
    Calculates and returns min, max, mean, std per field,
    which can be used to generate normalisation parameters.

    These are done via the data loading routines, which is
    slightly inefficient.
    '''
    dates = get_dates(year, start_hour=0, end_hour=168)

    mi = 0.0
    mx = 0.0
    dsum = 0.0
    dsqrsum = 0.0
    nsamples = 0
    for datestr in dates:
        for time_idx in range(28):
            data = load_fcst(field, datestr, time_idx)[:, :, 0]
            mi = min(mi, data.min())
            mx = max(mx, data.max())
            dsum += np.mean(data)
            dsqrsum += np.mean(np.square(data))
            nsamples += 1
    mn = dsum / nsamples
    sd = (dsqrsum/nsamples - mn**2)**0.5
    return mi, mx, mn, sd


def get_fcst_stats_fast(field, year=2018):
    '''
    Calculates and returns min, max, mean, std per field,
    which can be used to generate normalisation parameters.

    These are done directly from the forecast netcdf file,
    which is somewhat faster, as long as it fits into memory.
    '''
    ds_path = os.path.join(FCST_PATH, str(year), f"{field}.nc")
    nc_file = nc.Dataset(ds_path, mode="r")

    if field in accumulated_fields:
        data = nc_file[f"{field}_mean"][:, :-1, :, :]  # last time_idx is full of zeros
    else:
        data = nc_file[f"{field}_mean"][:, :, :, :]

    nc_file.close()

    if field in ["tp", "cp"]:
        # precip is measured in metres, so multiply to get mm
        data *= 1000
        data /= HOURS  # convert to mm/hr
        data = np.maximum(data, 0.0)  # shouldn't be necessary, but just in case
    elif field in accumulated_fields:
        # for all other accumulated fields [just ssr for us]
        data /= (HOURS*3600)  # convert from a 6-hr difference to a per-second rate

    mi = data.min()
    mx = data.max()
    mn = np.mean(data, dtype=np.float64)
    sd = np.std(data, dtype=np.float64)
    return mi, mx, mn, sd


def gen_fcst_norm(year=2018):
    '''
    One-off function, used to generate normalisation constants, which
    are used to normalise the various input fields for training/inference.
    '''

    stats_dic = {}
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")

    # make sure we can actually write there, before doing computation!!!
    with open(fcstnorm_path, 'wb') as f:
        pickle.dump(stats_dic, f)

    for field in all_fcst_fields:
        print(field)
        mi, mx, mn, sd = get_fcst_stats_fast(field, year)
        stats_dic[field] = {}
        stats_dic[field]['min'] = mi
        stats_dic[field]['max'] = mx
        stats_dic[field]['mean'] = mn
        stats_dic[field]['std'] = sd

    with open(fcstnorm_path, 'wb') as f:
        pickle.dump(stats_dic, f)


def load_fcst_norm(year=2018):
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")
    with open(fcstnorm_path, 'rb') as f:
        return pickle.load(f)


try:
    fcst_norm = load_fcst_norm(2018)
except:  # noqa
    fcst_norm = None
