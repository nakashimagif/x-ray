import numpy as np
import os
from keras_preprocessing.image import (ImageDataGenerator, Iterator,
                                       _list_valid_filenames_in_directory,
                                       load_img, img_to_array)

class myImageDataGenerator(ImageDataGenerator):
    def flow_from_dataframe(self, dataframe, directory,
                            x_col="filename", y_col="class", has_ext=True,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            subset=None,
                            interpolation='nearest'):
        return myDataFrameIterator(dataframe, directory, self,
                                 x_col=x_col, y_col=y_col, has_ext=has_ext,
                                 target_size=target_size, color_mode=color_mode,
                                 classes=classes, class_mode=class_mode,
                                 data_format=self.data_format,
                                 batch_size=batch_size, shuffle=shuffle, seed=seed,
                                 save_to_dir=save_to_dir,
                                 save_prefix=save_prefix,
                                 save_format=save_format,
                                 subset=subset,
                                 interpolation=interpolation)

# DSTF6:
# Modify DataFrameIterator in keras_preprocessing/image.py
# to support multi-label model.
class myDataFrameIterator(Iterator):
    def __init__(self, dataframe, directory, image_data_generator,
                 x_col="filenames", y_col="class", has_ext=True,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        super(myDataFrameIterator, self).common_init(image_data_generator,
                                                   target_size,
                                                   color_mode,
                                                   data_format,
                                                   save_to_dir,
                                                   save_prefix,
                                                   save_format,
                                                   subset,
                                                   interpolation)
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('Install pandas to use flow_from_dataframe.')
        if type(x_col) != str:
            raise ValueError("x_col must be a string.")
        if type(has_ext) != bool:
            raise ValueError("has_ext must be either True if filenames in"
                             " x_col has extensions,else False.")
        self.df = dataframe.drop_duplicates(x_col)
        self.df[x_col] = self.df[x_col].astype(str)
        self.directory = directory
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', 'other', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             '"other" or None.')
        self.class_mode = class_mode
        self.dtype = dtype
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                              'ppm', 'tif', 'tiff'}
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            if class_mode not in ["other", "input", None]:
                classes = list(self.df[y_col].unique())
        else:
            if class_mode in ["other", "input", None]:
                raise ValueError('classes cannot be set if class_mode'
                                 ' is either "other" or "input" or None.')
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        # Second, build an index of the images.
        self.filenames = []
        filenames = _list_valid_filenames_in_directory(
            directory,
            white_list_formats,
            self.split,
            class_indices=self.class_indices,
            follow_links=follow_links,
            df=True)
        if has_ext:
            ext_exist = False
            for ext in white_list_formats:
                if self.df.loc[0, x_col].endswith("." + ext):
                    ext_exist = True
                    break
            if not ext_exist:
                raise ValueError('has_ext is set to True but'
                                 ' extension not found in x_col')
            self.df = self.df[self.df[x_col].isin(filenames)].sort_values(by=x_col)
            self.filenames = list(self.df[x_col])
        else:
            without_ext_with = {f[:-1 * (len(f.split(".")[-1]) + 1)]: f
                                for f in filenames}
            filenames_without_ext = [f[:-1 * (len(f.split(".")[-1]) + 1)]
                                     for f in filenames]
            self.df = (self.df[self.df[x_col].isin(filenames_without_ext)]
                       .sort_values(by=x_col))
            self.filenames = [without_ext_with[f] for f in list(self.df[x_col])]
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype='int32')
        if class_mode not in ["other", "input", None]:
            self.classes = self.df[y_col].values
            #self.classes = np.array([self.class_indices[cls] for cls in classes])
        elif class_mode == "other":
            self.data = self.df[y_col].values
            if type(y_col) == str:
                y_col = [y_col]
            if "object" in list(self.df[y_col].dtypes):
                raise TypeError("y_col column/s must be numeric datatypes.")
        if self.num_classes > 0:
            print('Found %d images belonging to %d classes.' %
                  (self.samples, self.num_classes))
        else:
            print('Found %d images.' % self.samples)

        super(myDataFrameIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=self.dtype)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(self.dtype)
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=self.dtype)
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'other':
            batch_y = self.data[index_array]
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)