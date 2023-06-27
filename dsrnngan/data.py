""" File for handling data loading and saving. """
import os
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

import read_config


data_paths = read_config.get_data_paths()
TRUTH_PATH = data_paths["GENERAL"]["TRUTH_PATH"]
FCST_PATH = data_paths["GENERAL"]["FORECAST_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]

all_fcst_fields = ['tp', 'cp', 'sp', 'tisr', 'cape', 'tclw', 'tcwv', 'u700', 'v700']
fcst_hours = np.array([i for i in range(5)] + [i for i in range(6, 17)] + [i for i in range(18, 24)])


def denormalise(x):
    """
    Undo log-transform of rainfall.  Also cap at 500 (feel free to adjust according to application!)
    """
    return np.minimum(10**x - 1, 500.0)


def get_dates(year, ecmwf_file_order=False):
    """
    Return dates where we have truth data
    """
    if ecmwf_file_order:
        # return dates in the order ECMWF machine returned them!
        if str(year) == "2019":
            return ['20190203', '20191105', '20190414', '20190502', '20190604', '20190722', '20191006', '20190306', '20190731', '20190202', '20190321', '20191231', '20190909', '20190523', '20191205', '20190512', '20190220', '20190712', '20191220', '20190706', '20191008', '20190617', '20191210', '20190513', '20190613', '20190429', '20190616', '20190317', '20190529', '20190406', '20190114', '20190806', '20191128', '20190804', '20190622', '20190830', '20190312', '20191022', '20190311', '20191108', '20191230', '20190805', '20190222', '20190304', '20190517', '20190603', '20190531', '20191029', '20190411', '20190921', '20190422', '20190212', '20191007', '20191004', '20191202', '20190803', '20190819', '20190815', '20191206', '20190131', '20190305', '20190218', '20190102', '20191020', '20190605', '20190217', '20190405', '20190525', '20190204', '20190919', '20191122', '20190701', '20191211', '20190127', '20191118', '20190729', '20190207', '20190430', '20191126', '20191225', '20191119', '20190201', '20190905', '20190825', '20191116', '20191013', '20190507', '20190125', '20190828', '20190504', '20190308', '20190409', '20191030', '20191023', '20190607', '20190618', '20190926', '20190104', '20191014', '20190715', '20190417', '20190526', '20191027', '20191017', '20190807', '20191121', '20190728', '20191001', '20190428', '20190116', '20191129', '20191227', '20190612', '20190412', '20190902', '20190119', '20190209', '20191209', '20190913', '20190810', '20190813', '20190827', '20190113', '20191019', '20190602', '20190629', '20190206', '20190720', '20190330', '20190506', '20190221', '20190126', '20190606', '20190205', '20190623', '20190717', '20190707', '20190107', '20190929', '20190910', '20191224', '20190709', '20191103', '20190322', '20190901', '20191024', '20191101', '20191207', '20191016', '20190615', '20191223', '20190106', '20190501', '20190408', '20190423', '20190619', '20191011', '20191218', '20190801', '20190105', '20191025', '20190505', '20190117', '20190918', '20190421', '20190620', '20190626', '20191228', '20190510', '20191012', '20191028', '20190824', '20190710', '20190821', '20190608', '20190226', '20190403', '20190128', '20191120', '20190816', '20190705', '20190120', '20190627', '20190410', '20190111', '20190516', '20190927', '20190614', '20190320', '20190904', '20190914', '20191219', '20191107', '20190823', '20190511', '20190923', '20190723', '20191115', '20190328', '20190916', '20190726', '20191112', '20190527', '20191222', '20190808', '20191018', '20190401', '20190703', '20190907', '20190425', '20191031', '20191010', '20190521', '20191123', '20190702', '20190228', '20190101', '20190215', '20191216', '20190302', '20190915', '20191111', '20190628', '20191002', '20190227', '20190524', '20190402', '20190424', '20190216', '20190911', '20190812', '20190621', '20191102', '20190727', '20191130', '20190210', '20190704', '20191106', '20190211', '20190331', '20190318', '20191201', '20190624', '20191021', '20191117', '20190522', '20190110', '20190314', '20190719', '20190418', '20191110', '20190518', '20190817', '20190112', '20190809', '20191215', '20190528', '20190630', '20191109', '20190912', '20191113', '20190326', '20190122', '20190625', '20191015', '20190413', '20190420', '20191104', '20191204', '20191203', '20190903', '20190327', '20190814', '20190315', '20190922', '20190811', '20191221', '20191214', '20190407', '20191125', '20190820', '20190724', '20190123', '20190103', '20190310', '20190416', '20190115', '20190826', '20190219', '20190917', '20190303', '20190906', '20191226', '20190118', '20190208', '20190601', '20190301', '20190121', '20190930', '20190323', '20190307', '20190725', '20190427', '20190404', '20190609', '20190108', '20190503', '20190515', '20190802', '20191003', '20190924', '20190324', '20191229', '20190730', '20190514', '20190309', '20190415', '20190329', '20191026', '20190708', '20190130', '20190530', '20190224', '20190124', '20190419', '20190319', '20190908', '20191208', '20190928', '20191005', '20190711', '20190829', '20190223', '20190313', '20191212', '20190508', '20190316', '20190831', '20191124', '20190920', '20190325', '20191213', '20191114', '20190109', '20190509', '20190822', '20190611', '20190129', '20190225', '20190610']
        elif str(year) == "2020":
            return ['20200814', '20200718', '20200125', '20200420', '20200521', '20200815', '20200208', '20200503', '20200624', '20200330', '20200802', '20200408', '20200724', '20201103', '20200705', '20201109', '20200614', '20200212', '20200509', '20201201', '20201012', '20200723', '20200416', '20200510', '20200213', '20201218', '20201206', '20200221', '20200725', '20201108', '20201117', '20200701', '20201106', '20200429', '20200328', '20201128', '20200620', '20200525', '20200413', '20200927', '20200923', '20201020', '20200823', '20201211', '20200806', '20201125', '20201010', '20200505', '20200531', '20200609', '20201231', '20200824', '20200303', '20201015', '20200111', '20201018', '20201019', '20200305', '20200229', '20200520', '20200114', '20201005', '20200830', '20200106', '20200124', '20200327', '20200112', '20200418', '20201113', '20200128', '20201031', '20200502', '20200422', '20201210', '20200504', '20200828', '20200829', '20200507', '20200917', '20200628', '20201209', '20201002', '20200913', '20200825', '20201204', '20200102', '20200915', '20200901', '20200810', '20201017', '20200423', '20200714', '20200225', '20200108', '20201107', '20201208', '20201110', '20200719', '20201028', '20201001', '20200630', '20200129', '20200607', '20200610', '20201025', '20200526', '20201229', '20200309', '20200406', '20201216', '20200808', '20201202', '20200519', '20200622', '20201127', '20200604', '20200428', '20200618', '20200530', '20201215', '20200721', '20201007', '20200617', '20200601', '20201022', '20200312', '20200907', '20200518', '20200105', '20200430', '20200615', '20200512', '20200710', '20200516', '20201130', '20201221', '20200409', '20200811', '20200219', '20200803', '20200817', '20200324', '20200703', '20201126', '20200411', '20201223', '20200104', '20201021', '20201116', '20200326', '20200805', '20200926', '20200223', '20200819', '20200818', '20201121', '20200228', '20200107', '20200816', '20200904', '20201004', '20200421', '20200627', '20200804', '20200827', '20200821', '20201105', '20200130', '20200425', '20201003', '20201205', '20201118', '20200501', '20200121', '20200911', '20201214', '20200323', '20200902', '20201026', '20200407', '20201212', '20200711', '20200404', '20200414', '20200905', '20200529', '20200523', '20200820', '20200813', '20201217', '20200403', '20200207', '20201203', '20200122', '20200918', '20200216', '20200809', '20200115', '20201213', '20200912', '20200920', '20201129', '20200712', '20201114', '20201222', '20200606', '20200706', '20200910', '20201119', '20200415', '20200206', '20200914', '20200602', '20200118', '20200702', '20200909', '20200603', '20200924', '20200807', '20200605', '20200515', '20201124', '20201023', '20200517', '20200506', '20201011', '20200826', '20200412', '20200227', '20201230', '20200822', '20201220', '20200417', '20201024', '20201226', '20200424', '20200117', '20200727', '20200623', '20200203', '20201207', '20201008', '20200929', '20200304', '20200801', '20200728', '20200611', '20200925', '20200524', '20200709', '20200214', '20201006', '20200906', '20201013', '20201122', '20200205', '20200402', '20200716', '20200204', '20200306', '20200426', '20200715', '20200616', '20200126', '20200522', '20201115', '20200101', '20200514', '20200201', '20200919', '20200302', '20200508', '20200612', '20200307', '20201014', '20200127', '20201111', '20200131', '20200120', '20200224', '20200908', '20200720', '20200209', '20200113', '20201029', '20200922', '20200626', '20200116', '20200218', '20200726', '20200619', '20201112', '20201009', '20201102', '20201104', '20200301', '20201016', '20201224', '20200405', '20200903', '20201219', '20201101', '20200831', '20200704', '20200123', '20200427', '20200621', '20200511', '20200513', '20200215', '20200211', '20200419', '20200629', '20200310', '20200930', '20201227', '20200329', '20200103', '20200217', '20200308', '20200410', '20201225', '20200226', '20200731', '20201120', '20200527', '20200916', '20201123', '20200625', '20200210', '20201030', '20200713', '20200613', '20201027', '20200717', '20200928', '20200707', '20200110', '20200528', '20200401', '20200109', '20200608', '20200331', '20201228', '20200812', '20200722', '20200119', '20200220', '20200921']
    from glob import glob
    file_paths = os.path.join(TRUTH_PATH, str(year), "*.nc")
    files = glob(file_paths)
    dates = []
    for f in files:
        dates.append(f[:-3].split('_')[-1])
    return sorted(dates)


def load_truth_and_mask(date, hour, log_precip=False, aggregate=1):
    year = date[:4]
    data_path = os.path.join(TRUTH_PATH, year, f"metoffice-c-band-rain-radar_uk_{date}.nc")
    data = xr.open_dataset(data_path)
    assert hour+aggregate < 25
    y = np.array(data['unknown'][hour:hour+aggregate, :, :]).sum(axis=0)
    data.close()
    # The remapping of the NIMROD radar left a few negative numbers, so remove those
    y[y < 0.0] = 0.0

    # crop from 951x951 down to 940x940
    y = y[5:-6, 5:-6]

    # mask: False for valid truth data, True for invalid truth data
    # (compatible with the NumPy masked array functionality)
    # if all data is valid:
    mask = np.full(y.shape, False, dtype=bool)

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


def load_fcst_truth_batch(batch_dates, fcst_fields=all_fcst_fields, log_precip=False,
                          hour=0, norm=False):
    batch_x = []  # forecast
    batch_y = []  # truth
    batch_mask = []  # mask

    if type(hour) == str:
        if hour == 'random':
            hours = fcst_hours[np.random.randint(len(fcst_hours), size=[len(batch_dates)])]
        else:
            assert False, f"Not configured for {hour}"
    elif np.issubdtype(type(hour), np.integer):
        hours = len(batch_dates)*[hour]
    else:
        hours = hour

    for i, date in enumerate(batch_dates):
        h = hours[i]
        batch_x.append(load_fcst_stack(fcst_fields, date, h, log_precip=log_precip, norm=norm))
        truth, mask = load_truth_and_mask(date, h, log_precip=log_precip)
        batch_y.append(truth)
        batch_mask.append(mask)

    return np.array(batch_x), np.array(batch_y), np.array(batch_mask)


def load_fcst(ifield, date, hour, log_precip=False, norm=False):
    # Get the time required (compensating for IFS forecast saving precip at the end of the timestep)
    time = datetime(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:8]), hour=hour) + timedelta(hours=1)

    # Get the correct forecast starttime
    if time.hour < 6:
        tmpdate = time - timedelta(days=1)
        loaddate = datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=18)
        loadtime = '12'
    elif 6 <= time.hour < 18:
        tmpdate = time
        loaddate = datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=6)
        loadtime = '00'
    elif 18 <= time.hour < 24:
        tmpdate = time
        loaddate = datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=18)
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


def load_fcst_stack(fields, date, hour, log_precip=False, norm=False):
    field_arrays = []
    for f in fields:
        field_arrays.append(load_fcst(f, date, hour, log_precip=log_precip, norm=norm))
    return np.stack(field_arrays, -1)


def get_fcst_stats(field, year=2018):
    import datetime

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
