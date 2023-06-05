""" File for handling data loading and saving. """
import os
import datetime

import numpy as np
import xarray as xr

import read_config


data_paths = read_config.get_data_paths()
RADAR_PATH = data_paths["GENERAL"]["RADAR_PATH"]
FCST_PATH = data_paths["GENERAL"]["FORECAST_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]

all_fcst_fields = ['tp', 'cp', 'sp', 'tisr', 'cape', 'tclw', 'tcwv', 'u700', 'v700']


def denormalise(x):
    """
    Undo log-transform of rainfall.  Also cap at 500 (feel free to adjust according to application!)
    """
    return np.minimum(10**x - 1, 500.0)


def get_dates(year,
              lead_time,
              length,
              start_time):
    '''
    Returns a list of valid dates for which sufficient radar data exists,
    given the other input parameters.  Radar data may not be available
    for certain days, hence this is not quite the full year.
    Parameters:
        year (int): forecasts starting in this year
        lead_time (int): Lead time of first forecast/truth pair in sequence
        length (int): Number of sequence steps to return (1+)
        start_time (string): Either '00' or '12', the IFS forecast to use
    '''
    assert start_time in ('00', '12')
    assert length >= 1
    assert lead_time + length <= 17, "only 17 hours of forecasts in this dataset"

    # Build "cache" of radar dates that exist
    radar_cache = set()
    date = datetime.date(year, 1, 1)
    oneday = datetime.timedelta(days=1)
    end_date = datetime.date(year+1, 1, 10)  # go a bit into following year since lead_time + length could be several days
    while date < end_date:
        datestr = date.strftime('%Y%m%d')
        if os.path.exists(os.path.join(RADAR_PATH, str(date.year), f"metoffice-c-band-rain-radar_uk_{datestr}.nc")):
            radar_cache.add(datestr)
        date += oneday

    # Now work out which IFS start dates to use. For each candidate start date,
    # work out which NIMROD dates are needed, and check if they exist.

    # TODO: check off-by-one errors due to accumulated fields, etc.

    # Could check the IFS file to be safer?  But would be a lot slower...
    if start_time == '00':
        ifsstart = datetime.datetime(year, 1, 1, hour=0)
    else:
        ifsstart = datetime.datetime(year, 1, 1, hour=12)

    end_date = datetime.datetime(year+1, 1, 1)
    valid_dates = []
    while ifsstart < end_date:
        # Check hour by hour.  Not particularly efficient, but almost certainly
        # not a bottleneck, since we don't hit the disk. Could re-write if needed.
        valid = True
        # check for off-by-one error here?
        for ii in range(lead_time, lead_time+length):
            nimtime = ifsstart + datetime.timedelta(hours=ii)
            if nimtime.strftime('%Y%m%d') not in radar_cache:
                valid = False
                break
        if valid:
            datestr = ifsstart.strftime('%Y%m%d')
            valid_dates.append(datestr)

        ifsstart += oneday
    return np.array(valid_dates)


def load_radar_and_mask(fcst_date, lead_time, length, start_time, log_precip=False, aggregate=1):
    # need to calcuate radar date[s] from input arguments
    # date, lead_time, length, start_time
    fcst_start = datetime.datetime.strptime(fcst_date, '%Y%m%d')
    if start_time == "00":
        fcst_start = fcst_start.replace(hour=0)
    elif start_time == "12":
        fcst_start = fcst_start.replace(hour=12)
    data_start = fcst_start + datetime.timedelta(hours=lead_time)

    year = date[:4]
    data_path = os.path.join(RADAR_PATH, year, f"metoffice-c-band-rain-radar_uk_{date}.nc")
    data = xr.open_dataset(data_path)
    assert hour+aggregate < 25
    y = np.array(data['unknown'][hour:hour+aggregate, :, :]).sum(axis=0)
    data.close()
    # The remapping of the NIMROD radar left a few negative numbers, so remove those
    y[y < 0.0] = 0.0

    # crop from 951x951 down to 940x940
    y = y[5:-6, 5:-6]

    # mask: False for valid radar data, True for invalid radar data
    # (compatible with the NumPy masked array functionality)
    # if all data is valid:
    # mask = np.full(y.shape, False, dtype=bool)
    mask = np.load("/ppdata/NIMROD_mask/original.npy")

    if log_precip:
        return np.log10(1+y), mask
    else:
        return y, mask


def logprec(y, log_precip=True):
    if log_precip:
        return np.log10(1+y)
    else:
        return y


def load_hires_constants(batch_size=1):
    lsm_path = os.path.join(CONSTANTS_PATH, "hgj2_constants_0.01_degree.nc")
    df = xr.load_dataset(lsm_path)
    # LSM is already 0:1
    lsm = np.array(df['LSM'])[:, ::-1, :]
    df.close()

    oro_path = os.path.join(CONSTANTS_PATH, "topo_local_0.01.nc")
    df = xr.load_dataset(oro_path)
    # Orography.  Clip below, to remove spectral artifacts, and normalise by max
    z = df['z'].data
    z = z[:, ::-1, :]
    z[z < 5] = 5
    z = z/z.max()

    df.close()
    # print(z.shape, lsm.shape)
    # crop from 951x951 down to 940x940
    lsm = lsm[..., 5:-6, 5:-6]
    z = z[..., 5:-6, 5:-6]
    return np.repeat(np.stack([z, lsm], -1), batch_size, axis=0)


# early attempt, maybe something useful in here?
# def get_seq_data(fcst_fields, start_date, start_hour, lead_time, length,
#                  log_precip=False, ifs_norm=False):
#     ifslong = xr.open_dataset(f'{IFS_PATH_FLOOD}/sfc_{start_date}_{start_hour}.nc')

#     # IFS labels time by the end of the hour rather than beginning
#     time = ifslong.time[lead_time] - pd.Timedelta(hours=1)

#     nim = load_nimrod_seq(time.dt.strftime('%Y%m%d').item(), time.dt.hour.item(),
#                           log_precip=log_precip, aggregate=1)

#     ifs = load_ifsstack_seq(fields, start_date, start_hour, lead_time,
#                             log_precip=log_precip, norm=ifs_norm)

#     ifslong.close()
#     return ifs, nim


def load_fcst_radar_batch(batch_dates,
                          lead_time,
                          length,
                          start_time,
                          fcst_fields=all_fcst_fields,
                          log_precip=False,
                          norm=False):
    '''
    Returns a temporal sequence of (forecast, truth, mask) data.
    Parameters:
        batch_dates (list of strings): Dates of forecasts
        lead_time (int): Lead time of first forecast/truth pair in sequence
        length (int): Number of sequence steps to return (1+)
        start_time (string): Start hour of forecast, either '00' or '12'
        fcst_fields (list of strings): The fields to be used
        log_precip (bool): Whether to apply log10(1+x) transform to precip-related fields
        ifs_norm (bool): Whether to apply normalisation to fields to make O(1)
    '''
    batch_x = []  # forecast
    batch_y = []  # radar
    batch_mask = []  # mask

    for fcst_date in batch_dates:
        batch_x.append(load_fcst_stack(fcst_fields, fcst_date, lead_time, length, start_time, log_precip=log_precip, norm=norm))
        radar, mask = load_radar_and_mask(fcst_date, lead_time, length, start_time, log_precip=log_precip)
        batch_y.append(radar)
        batch_mask.append(mask)
    return np.array(batch_x), np.array(batch_y), np.array(batch_mask)


def load_fcst(ifield, date, lead_time, length, start_time, log_precip=False, norm=False):
    # Get the time required (compensating for IFS forecast saving precip at the end of the timestep)
    time = datetime.datetime(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:8]), hour=hour) + datetime.timedelta(hours=1)

# REWRITE ALL THIS
    # Get the correct forecast starttime
    if time.hour < 6:
        tmpdate = time - datetime.timedelta(days=1)
        loaddate = datetime.datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=18)
        loadtime = '12'
    elif 6 <= time.hour < 18:
        tmpdate = time
        loaddate = datetime.datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=6)
        loadtime = '00'
    elif 18 <= time.hour < 24:
        tmpdate = time
        loaddate = datetime.datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=18)
        loadtime = '12'
    else:
        assert False, "Not acceptable time"
    dt = time - loaddate
    time_index = int(dt.total_seconds()//3600)
    assert time_index >= 1, "Cannot use first hour of retrival"
    loaddata_str = loaddate.strftime("%Y%m%d") + '_' + loadtime

    field = ifield
    if field in ['u700', 'v700']:
        fleheader = 'winds'
        field = field[:1]
    elif field in ['cdir', 'tcrw']:
        fleheader = 'missing'
    else:
        fleheader = 'sfc'

    ds_path = os.path.join(FCST_PATH, f"{fleheader}_{loaddata_str}.nc")
    ds = xr.open_dataset(ds_path)
    data = ds[field]
    field = ifield
    if field in ['tp', 'cp', 'cdir', 'tisr']:
        data = data[time_index, :, :] - data[time_index-1, :, :]
    else:
        data = data[time_index, :, :]

    y = np.array(data[::-1, :])
    # crop from 96x96 to 94x94
    y = y[1:-1, 1:-1]
    data.close()
    ds.close()
    if field in ['tp', 'cp', 'pr', 'prl', 'prc']:
        # print('pass')
        y[y < 0] = 0.
        y = 1000*y
    if log_precip and field in ['tp', 'cp', 'pr', 'prc', 'prl']:
        # precip is measured in metres, so multiply up
        return np.log10(1+y)  # *1000)
    elif norm:
        return (y-fcst_norm[field][0])/fcst_norm[field][1]
    else:
        return y


def load_fcst_stack(fields, date, lead_time, length, start_time, log_precip=False, norm=False):
    field_arrays = []
    for f in fields:
        field_arrays.append(load_fcst(f, date, lead_time, length, start_time, log_precip=log_precip, norm=norm))
    return np.stack(field_arrays, -1)


def get_fcst_stats(field, year=2018):
    # create date objects
    begin_year = datetime.date(year, 1, 1)
    end_year = datetime.date(year, 12, 31)
    one_day = datetime.timedelta(days=1)
    next_day = begin_year

    mi = 0
    mx = 0
    mn = 0
    sd = 0
    nsamples = 0
    for day in range(0, 366):  # includes potential leap year
        if next_day > end_year:
            break
        for hour in fcst_hours:
            try:
                dta = load_fcst(field, next_day.strftime("%Y%m%d"), hour)
                mi = min(mi, dta.min())
                mx = max(mx, dta.max())
                mn += dta.mean()
                sd += dta.std()**2
                nsamples += 1
            except:  # noqa
                print(f"Problem loading {next_day.strftime('%Y%m%d')}, {hour}")
        next_day += one_day
    mn /= nsamples
    sd = (sd / nsamples)**0.5
    return mi, mx, mn, sd


def gen_fcst_norm(year=2018):

    """
    One-off function, used to generate normalisation constants, which are used to normalise the various input fields for training/inference.

    Depending on the field, we may subtract the mean and divide by the std. dev., or just divide by the max observed value.
    """

    import pickle
    stats_dic = {}
    for f in all_fcst_fields:
        stats = get_fcst_stats(f, year)
        if f == 'sp':
            stats_dic[f] = [stats[2], stats[3]]
        elif f == "u700" or f == "v700":
            stats_dic[f] = [0, max(-stats[0], stats[1])]
        else:
            stats_dic[f] = [0, stats[1]]
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")
    with open(fcstnorm_path, 'wb') as f:
        pickle.dump(stats_dic, f, 0)
    return


def load_fcst_norm(year=2018):
    import pickle
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")
    with open(fcstnorm_path, 'rb') as f:
        return pickle.load(f)


try:
    fcst_norm = load_fcst_norm(2018)
except:  # noqa
    fcst_norm = None
