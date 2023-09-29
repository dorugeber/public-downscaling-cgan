""" Data generator class for full-image evaluation of precipitation downscaling network """
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from data import load_fcst_truth_batch, load_hires_constants, HOURS
import read_config


class DataGenerator(Sequence):
    '''
    Data generator class that returns (forecast, constants, mask, truth) data. Class will return forecast data at the start and end of each interval (for non-accumulated fields) and accumulated fields over the interval.  The truth data is averaged over the interval.

    DataGenerator(["20180409", "20200607"], fcst_fields=["cape", "tp"], start_hour=12, end_hour=24) will return data over two periods: 12-18 and 18-24 hours for the forecasts initialised on 20180409 and 20200607.
    '''
    def __init__(self, dates, fcst_fields,
                 start_hour=0, end_hour=168,
                 batch_size=1, log_precip=True,
                 shuffle=True, constants=True, fcst_norm=True,
                 autocoarsen=False, seed=9999):
        '''
        Forecast: input forecast data
        Constants: geographic fields; LSM and orography
        Mask: False where truth data is valid, True where truth data is invalid
        Truth: precipitation data
        Parameters:
            dates (list of YYYYMMDD strings): The forecast start dates to be used
            fcst_fields (list of strings): The forecast fields to be used
            start_hour (int): Lead time of first forecast/truth hour to use
            end_hour (int): Lead time of last forecast/truth hour to use
            batch size (int): Batch size
            log_precip (bool): Whether to apply log10(1+x) transform to precip-related fields
            shuffle (bool): Whether to shuffle data (else return sorted by date then lead time)
            constants (bool): Whether to return orography/LSM fields
            fcst_norm (bool): Whether to apply normalisation to fields to make O(1)
            autocoarsen (bool): Whether to replace forecast data by coarsened truth
            seed (int): Random seed given to NumPy, used for repeatable shuffles
        '''

        # sanity checks for our dataset
        assert start_hour >= 0
        assert end_hour <= 168
        assert start_hour % HOURS == 0
        assert end_hour % HOURS == 0
        assert end_hour >= start_hour + 24
        assert autocoarsen is False  # untested, probably not useful in this project

        self.fcst_fields = fcst_fields
        self.batch_size = batch_size
        self.log_precip = log_precip
        self.shuffle = shuffle
        self.fcst_norm = fcst_norm
        self.autocoarsen = autocoarsen
        self.seed = seed

        if self.autocoarsen:
            # read downscaling factor from file
            df_dict = read_config.read_downscaling_factor()  # read downscaling params
            self.ds_factor = df_dict["downscaling_factor"]

        if constants:
            self.constants = load_hires_constants(self.batch_size)
        else:
            self.constants = None

        # convert to numpy array for easy use of np.repeat
        temp_dates = np.array(dates)

        # represent valid lead-time intervals, 0 = 0-24 hours, 1 = 6-30 hours, 2 = 12-36 hours etc
        temp_time_idxs = np.arange(start_hour//HOURS, end_hour//HOURS - 3)

        # if no shuffle, the DataGenerator will return each interval from the
        # first date, then each interval from the second date, etc.
        self.dates = np.repeat(temp_dates, len(temp_time_idxs))
        self.time_idxs = np.tile(temp_time_idxs, len(temp_dates))

        if self.shuffle:
            rng = np.random.default_rng(seed)
            self.shuffle_data(rng)

    def __len__(self):
        # Number of batches in dataset
        return len(self.dates) // self.batch_size

    def _dataset_autocoarsener(self, truth):
        kernel_tf = tf.constant(1.0/(self.ds_factor*self.ds_factor), shape=(self.ds_factor, self.ds_factor, 1, 1), dtype=tf.float32)
        image = tf.nn.conv2d(truth, filters=kernel_tf, strides=[1, self.ds_factor, self.ds_factor, 1], padding='VALID',
                             name='conv_debug', data_format='NHWC')
        return image

    def __getitem__(self, idx):
        # Get batch at index idx
        dates_batch = self.dates[idx*self.batch_size:(idx+1)*self.batch_size]
        time_idx_batch = self.time_idxs[idx*self.batch_size:(idx+1)*self.batch_size]

        # Load and return this batch of data
        data_x_batch, data_y_batch, data_mask_batch = load_fcst_truth_batch(
            dates_batch,
            time_idx_batch,
            fcst_fields=self.fcst_fields,
            log_precip=self.log_precip,
            norm=self.fcst_norm)

        if self.autocoarsen:
            # replace forecast data by coarsened truth data!
            truth_temp = data_y_batch.copy()
            truth_temp[data_mask_batch] = 0.0
            data_x_batch = self._dataset_autocoarsener(truth_temp[..., np.newaxis])

        if self.constants is None:
            return {"lo_res_inputs": data_x_batch},\
                   {"output": data_y_batch,
                    "mask": data_mask_batch}
        else:
            return {"lo_res_inputs": data_x_batch,
                    "hi_res_inputs": self.constants},\
                   {"output": data_y_batch,
                    "mask": data_mask_batch}

    def shuffle_data(self, rng):
        assert len(self.time_idxs) == len(self.dates)
        # shuffle both dates and time index arrays the same way
        p = rng.permutation(len(self.dates))
        self.dates = self.dates[p]
        self.time_idxs = self.time_idxs[p]

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()


if __name__ == "__main__":
    pass
