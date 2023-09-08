import glob
import os
import random

import numpy as np
import tensorflow as tf

import read_config
from data import all_fcst_fields, denormalise, get_dates, HOURS


data_paths = read_config.get_data_paths()
records_folder = data_paths["TFRecords"]["tfrecords_path"]
ds_fac = read_config.read_downscaling_factor()["downscaling_factor"]

CLASSES = 4
SEQ_LENGTH = 8  # train on sequences of length 8 - this is what Leinonen used, may wish to vary in the future!
DEFAULT_FCST_SHAPE = (SEQ_LENGTH, 64, 64, 4*len(all_fcst_fields))
DEFAULT_CON_SHAPE = (SEQ_LENGTH, 64, 64, 2)
DEFAULT_OUT_SHAPE = (SEQ_LENGTH, 64, 64, 1)


def DataGenerator(years, batch_size, repeat=True, autocoarsen=False, weights=None):
    return create_mixed_dataset(years, batch_size, repeat=repeat, autocoarsen=autocoarsen, weights=weights)


def create_mixed_dataset(years,
                         batch_size,
                         fcst_shape=DEFAULT_FCST_SHAPE,
                         con_shape=DEFAULT_CON_SHAPE,
                         out_shape=DEFAULT_OUT_SHAPE,
                         repeat=True,
                         autocoarsen=False,
                         folder=records_folder,
                         shuffle_size=64,
                         weights=None):

    if weights is None:
        weights = [1./CLASSES]*CLASSES
    datasets = [create_dataset(years,
                               ii,
                               fcst_shape=fcst_shape,
                               con_shape=con_shape,
                               out_shape=out_shape,
                               folder=folder,
                               shuffle_size=shuffle_size,
                               repeat=repeat)
                for ii in range(CLASSES)]
    sampled_ds = tf.data.Dataset.sample_from_datasets(datasets,
                                                      weights=weights).batch(batch_size)

    if autocoarsen:
        sampled_ds = sampled_ds.map(_dataset_autocoarsener)
    sampled_ds = sampled_ds.prefetch(2)
    return sampled_ds

# Note, if we wanted fewer classes, we can use glob syntax to grab multiple classes as once
# e.g. create_dataset(2015,"[67]")
# will take classes 6 & 7 together


def _dataset_autocoarsener(inputs, outputs):
    image = outputs['output']
    kernel_tf = tf.constant(1.0/(ds_fac*ds_fac), shape=(ds_fac, ds_fac, 1, 1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, ds_fac, ds_fac, 1], padding='VALID',
                         name='conv_debug', data_format='NHWC')
    inputs['lo_res_inputs'] = image
    return inputs, outputs


def _parse_batch(record_batch,
                 insize=DEFAULT_FCST_SHAPE,
                 consize=DEFAULT_CON_SHAPE,
                 outsize=DEFAULT_OUT_SHAPE):
    # Create a description of the features
    feature_description = {
        'generator_input': tf.io.FixedLenFeature(insize, tf.float32),
        'constants': tf.io.FixedLenFeature(consize, tf.float32),
        'generator_output': tf.io.FixedLenFeature(outsize, tf.float32),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    return ({'lo_res_inputs': example['generator_input'],
             'hi_res_inputs': example['constants']},
            {'output': example['generator_output']})


def create_dataset(years,
                   clss,
                   fcst_shape=DEFAULT_FCST_SHAPE,
                   con_shape=DEFAULT_CON_SHAPE,
                   out_shape=DEFAULT_OUT_SHAPE,
                   folder=records_folder,
                   shuffle_size=64,
                   repeat=True):
    # TODO: tf.data.Dataset.list_files should accept the list of glob patterns,
    # not the list of globbed filenames

    # "The file_pattern argument should be a small number of glob patterns. If your
    # filenames have already been globbed, use Dataset.from_tensor_slices(filenames)
    # instead, as re-globbing every filename with list_files may result in poor
    # performance with remote storage systems."

    # however, tried this on EWC and it was marginally slower!
    # But may want to change in future
    filelist = []
    for yr in years:
        fpattern = os.path.join(folder, f"{yr}_*.{clss}.tfrecords")
        filelist += glob.glob(fpattern)

    files_ds = tf.data.Dataset.list_files(filelist)
    ds = tf.data.TFRecordDataset(files_ds,
                                 compression_type="GZIP",
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.shuffle(shuffle_size)
    ds = ds.map(lambda x: _parse_batch(x,
                                       insize=fcst_shape,
                                       consize=con_shape,
                                       outsize=out_shape))
    if repeat:
        return ds.repeat()
    else:
        return ds


# create_fixed_dataset currently unused; full image dataset used for validation
def create_fixed_dataset(year=None,
                         mode='validation',
                         batch_size=16,
                         autocoarsen=False,
                         fcst_shape=DEFAULT_FCST_SHAPE,
                         con_shape=DEFAULT_CON_SHAPE,
                         out_shape=DEFAULT_OUT_SHAPE,
                         name=None,
                         folder=records_folder):
    assert year is not None or name is not None, "Must specify year or file name"
    if name is None:
        name = os.path.join(folder, f"{mode}{year}.tfrecords")
    else:
        name = os.path.join(folder, name)
    fl = glob.glob(name)
    files_ds = tf.data.Dataset.list_files(fl)
    ds = tf.data.TFRecordDataset(files_ds,
                                 num_parallel_reads=1)
    ds = ds.map(lambda x: _parse_batch(x,
                                       insize=fcst_shape,
                                       consize=con_shape,
                                       outsize=out_shape))
    ds = ds.batch(batch_size)
    if autocoarsen:
        ds = ds.map(_dataset_autocoarsener)
    return ds


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def write_data(year,
               folder=records_folder,
               fcst_fields=all_fcst_fields,
               img_chunk_width=DEFAULT_FCST_SHAPE[1],  # controls size of subsampled image
               num_class=CLASSES,
               log_precip=True,
               fcst_norm=True):
    from data_generator import DataGenerator as DataGeneratorFull
    assert isinstance(year, int)

    # change this to your forecast image size!
    img_size_h = 384
    img_size_w = 352

    # binning: bin 0 is sample mean rainfall < 0.2mm/hr, bin 1 is 0.2-0.3mm/hr, etc
    bins = [0.2, 0.3, 0.45]  # not updated for multiple frames yet
    assert num_class == 4

    scaling_factor = ds_fac

    # chosen to approximately cover the full image, but can be changed!
    nsamples = 1  # img_size_h*img_size_w//(img_chunk_width**2)
    print("Samples per image:", nsamples)  # note, actual samples may be less than this if mask is used to exclude some

    # split TFRecords by lead time, in case this is useful for training on subsets of lead time
    for time_idx in range(28 + 1 - SEQ_LENGTH):
        print(f"Doing time index {time_idx}")
        s_hour = time_idx*HOURS
        e_hour = (time_idx + SEQ_LENGTH)*HOURS
        dates = get_dates(year,
                          start_hour=s_hour,
                          end_hour=e_hour)
        dgc = DataGeneratorFull(dates,
                                fcst_fields=fcst_fields,
                                start_hour=s_hour,
                                end_hour=e_hour,
                                seq_length=SEQ_LENGTH,
                                batch_size=1,
                                log_precip=log_precip,
                                shuffle=True,
                                constants=True,
                                fcst_norm=fcst_norm,
                                seed=9999+time_idx)

        fle_hdles = []
        for fh in range(num_class):
            flename = os.path.join(folder, f"{year}_{time_idx}.{fh}.tfrecords")
            # compress generated TFRecords, courtesy Fenwick
            options = tf.io.TFRecordOptions(compression_type="GZIP")
            fle_hdles.append(tf.io.TFRecordWriter(flename, options=options))

        for batch in range(len(dgc)):
            if (batch % 10) == 0:
                print(time_idx, batch, len(dgc))
            sample = dgc.__getitem__(batch)

            for ii in range(nsamples):
                # e.g. for image width 94 and img_chunk_width 20, can have 0:20 up to 74:94
                idh = random.randint(0, img_size_h-img_chunk_width)
                idw = random.randint(0, img_size_w-img_chunk_width)

                mask = sample[1]['mask'][0,
                                         :,
                                         idh*scaling_factor:(idh+img_chunk_width)*scaling_factor,
                                         idw*scaling_factor:(idw+img_chunk_width)*scaling_factor].flatten()
                if np.any(mask):
                    # some of the truth data is invalid, so don't use this subsample
                    continue

                truth = sample[1]['output'][0,
                                            :,
                                            idh*scaling_factor:(idh+img_chunk_width)*scaling_factor,
                                            idw*scaling_factor:(idw+img_chunk_width)*scaling_factor].flatten()
                const = sample[0]['hi_res_inputs'][0,
                                                   :,
                                                   idh*scaling_factor:(idh+img_chunk_width)*scaling_factor,
                                                   idw*scaling_factor:(idw+img_chunk_width)*scaling_factor,
                                                   :].flatten()
                forecast = sample[0]['lo_res_inputs'][0,
                                                      :,
                                                      idh:idh+img_chunk_width,
                                                      idw:idw+img_chunk_width,
                                                      :].flatten()
                feature = {
                    'generator_input': _float_feature(forecast),
                    'constants': _float_feature(const),
                    'generator_output': _float_feature(truth)
                }
                features = tf.train.Features(feature=feature)
                example = tf.train.Example(features=features)
                example_to_string = example.SerializeToString()

                # decide which bin to put this sample in
                truth_raw = denormalise(truth)  # undo log10(1+x) transformation
                truth_mean = truth_raw.mean()
                if truth_mean < bins[0]:
                    clss = 0
                elif truth_mean < bins[1]:
                    clss = 1
                elif truth_mean < bins[2]:
                    clss = 2
                else:
                    clss = 3

                fle_hdles[clss].write(example_to_string)

        for fh in fle_hdles:
            fh.close()


# currently unused; was previously used to make small-image validation dataset,
# but this is now obsolete
def save_dataset(tfrecords_dataset, flename, max_batches=None):
    flename = os.path.join(records_folder, flename)
    fle_hdle = tf.io.TFRecordWriter(flename)
    for ii, sample in enumerate(tfrecords_dataset):
        print(ii)
        if max_batches is not None:
            if ii == max_batches:
                break
        for k in range(sample[1]['output'].shape[0]):
            feature = {
                'generator_input': _float_feature(sample[0]['lo_res_inputs'][k, ...].numpy().flatten()),
                'constants': _float_feature(sample[0]['hi_res_inputs'][k, ...].numpy().flatten()),
                'generator_output': _float_feature(sample[1]['output'][k, ...].numpy().flatten())
            }
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            example_to_string = example.SerializeToString()
            fle_hdle.write(example_to_string)
    fle_hdle.close()
    return
