""" Data generator class for full-image evaluation of precipitation downscaling network """
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from data import get_dates, load_fcst_radar_batch, load_hires_constants
import read_config


class DataGenerator(Sequence):
    '''
    Data generator class that returns a temporal sequence of (forecast, constants, mask, truth) data
    '''
    def __init__(self, year, lead_time, length, start_time,
                 fcst_fields, batch_size, log_precip=True,
                 shuffle=True, constants=None, fcst_norm=True,
                 autocoarsen=False, seed=9999):

        '''
        Forecast: (low-resolution) input forecast data
        Constants: (full-resolution) geographic fields; LSM and orography
        Mask: False where truth data is valid, True where truth data is invalid
        Truth: (full-resolution) precipitation data
        Parameters:
            year (int): IFS forecasts starting in this year
            lead_time (int): Lead time of first forecast/truth pair in sequence
            length (int): Number of sequence steps to return (1+)
            start_time (string): Either '00' or '12', the IFS forecast to use
            fcst_fields (list of strings): The forecast fields to be used
            batch size (int): Batch size
            log_precip (bool): Whether to apply log10(1+x) transform to precip-related fields
            shuffle (bool): Whether to shuffle valid dates
            constants (bool): Whether to return orography/LSM fields
            fcst_norm (bool): Whether to apply normalisation to fields to make O(1)
            autocoarsen (bool): Whether to replace forecast data by coarsened truth
            seed (int): Random seed given to NumPy, used for repeatable shuffles
        '''

        assert start_time in ('00', '12')  # IFS-specific sanity check
        assert length >= 1
        assert lead_time >= 7  # this dataset only has forecast data for 7-17hrs
        assert lead_time + length <= 17
        assert autocoarsen == False  # not implemented yet

        self.year = year
        self.lead_time = lead_time
        self.length = length
        self.start_time = start_time
        self.fcst_fields = fcst_fields
        self.batch_size = batch_size
        self.log_precip = log_precip
        self.shuffle = shuffle
        self.fcst_norm = fcst_norm
        self.autocoarsen = autocoarsen

        if constants:
            self.constants = load_hires_constants(self.batch_size)
        else:
            self.constants = None

        # get valid dates, where radar data exists
        self.dates = get_dates(year, lead_time, length, start_time)

        if self.shuffle:
            np.random.seed(seed)
            self.shuffle_data()

        if self.autocoarsen:
            # read downscaling factor from file
            df_dict = read_config.read_downscaling_factor()  # read downscaling params
            self.ds_factor = df_dict["downscaling_factor"]


    def __len__(self):
        # Number of batches in dataset
        return len(self.dates) // self.batch_size

#     def _dataset_autocoarsener(self, radar):
#         kernel_tf = tf.constant(1.0/(self.ds_factor*self.ds_factor), shape=(self.ds_factor, self.ds_factor, 1, 1), dtype=tf.float32)
#         image = tf.nn.conv2d(radar, filters=kernel_tf, strides=[1, self.ds_factor, self.ds_factor, 1], padding='VALID',
#                              name='conv_debug', data_format='NHWC')
#         return image

    def __getitem__(self, idx):
        # Get batch at index idx
        dates_batch = self.dates[idx*self.batch_size:(idx+1)*self.batch_size]

        # Load and return this batch of frames
        data_x_batch, data_y_batch, data_mask_batch = load_fcst_radar_batch(
            dates_batch,
            lead_time,
            length,
            start_time,
            fcst_fields=self.fcst_fields,
            log_precip=self.log_precip,
            hour=hours_batch,
            norm=self.fcst_norm)

#         if self.autocoarsen:
#             # replace forecast data by coarsened radar data!
#             radar_temp = data_y_batch.copy()
#             radar_temp[data_mask_batch] = 0.0
#             data_x_batch = self._dataset_autocoarsener(radar_temp[..., np.newaxis])

        if self.constants is None:
            return {"lo_res_inputs": data_x_batch},\
                   {"output": data_y_batch,
                    "mask": data_mask_batch}
        else:
            return {"lo_res_inputs": data_x_batch,
                    "hi_res_inputs": self.constants},\
                   {"output": data_y_batch,
                    "mask": data_mask_batch}

    def shuffle_data(self):
        p = np.random.permutation(len(self.dates))
        self.dates = self.dates[p]
        return

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()


if __name__ == "__main__":
    pass
