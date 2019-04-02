import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
from keras import callbacks
from keras_preprocessing.image import (array_to_img, img_to_array, load_img)

import myGenerator
from myMetrics import tf_auc, recall, precision, fscore

import cv2


def create_generator(horizontal_flip, dataframe, directory, y_col, batch_size,
                     shuffle):
    datagen = myGenerator.myImageDataGenerator(
        rescale=1. / 255, horizontal_flip=horizontal_flip)
    generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        x_col='Image Index',
        y_col=y_col,
        has_ext=True,
        target_size=(224, 224),
        color_mode='rgb',
        classes=[0, 1],
        class_mode='binary',
        batch_size=batch_size,
        shuffle=shuffle,
        seed=None,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        subset=None,
        interpolation='nearest')
    return generator


class trainer:
    def __init__(self, trainer_name, dataframe, data_proportion, y_col, lr,
                 batch_size, initial_epoch, epochs):
        self.trainer_name = trainer_name
        self.data_proportion = data_proportion
        self.y_col = y_col
        self.lr = lr
        self.batch_size = batch_size
        self.initial_epoch = initial_epoch
        self.epochs = epochs

        self.train_df = dataframe.query('split == "train"').sample(
            frac=data_proportion)
        self.validation_df = dataframe.query('split == "validation"').sample(
            frac=data_proportion)
        self.test_df = dataframe.query('split == "test"')
        self.train_df.reset_index(drop=True, inplace=True)
        self.validation_df.reset_index(drop=True, inplace=True)
        self.test_df.reset_index(drop=True, inplace=True)

        self.train_generator = create_generator(
            horizontal_flip=True,
            dataframe=self.train_df,
            directory='images',
            y_col=self.y_col,
            batch_size=self.batch_size,
            shuffle=True)

        self.validation_generator = create_generator(
            horizontal_flip=False,
            dataframe=self.validation_df,
            directory='images',
            y_col=self.y_col,
            batch_size=self.batch_size,
            shuffle=True)

        self.test_generator = create_generator(
            horizontal_flip=False,
            dataframe=self.test_df,
            directory='images',
            y_col=self.y_col,
            batch_size=self.batch_size,
            shuffle=False)

        self.train_steps_per_epoch = np.math.ceil(
            self.train_generator.n / self.batch_size)
        self.validation_steps_per_epoch = np.math.ceil(
            self.validation_generator.n / self.batch_size)
        self.test_steps_per_epoch = np.math.ceil(
            self.test_generator.n / self.batch_size)

        train_df_total = self.train_df.shape[0]
        self.class_weight_list = []
        for num in self.train_df[self.y_col].sum():
            train_df_neg = train_df_total - num
            weight = {
                0: num / train_df_total,
                1: train_df_neg / train_df_total
            }
            self.class_weight_list.append(weight)

        #Define Callbacks
        tb_cb = callbacks.TensorBoard(log_dir="logs/", histogram_freq=0)
        cp_cb = callbacks.ModelCheckpoint(
            filepath='logs/' + self.trainer_name + '_best.h5',
            monitor='val_auc',
            verbose=1,
            save_best_only=True,
            mode='max')
        cl_cb = callbacks.CSVLogger(
            'logs/training_log_' + self.trainer_name + '.csv',
            separator='\t',
            append=False)
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            verbose=1,
            patience=5,
            mode='max',
            min_delta=0.001)
        self.cb = [tb_cb, cp_cb, cl_cb, reduce_lr]

    def training(self, lr_search=False, **lr_search_dict):
        #Define Callbacks
        if lr_search is True:
            cb = []
            lr = lr_search_dict['lr']
            initial_epoch = 0
            epochs = 1
        else:
            cb = self.cb
            lr = self.lr
            initial_epoch = self.initial_epoch
            epochs = self.epochs

        #http://marubon-ds.blogspot.com/2017/10/inceptionv3-fine-tuning-model.html
        from keras.applications.densenet import DenseNet121
        input_tensor = layers.Input(shape=(224, 224, 3))
        inc_model = DenseNet121(
            input_tensor=input_tensor, weights='imagenet', include_top=False)

        # get layers and add average pooling layer
        x = inc_model.output
        x = layers.GlobalAveragePooling2D()(x)

        # add output layer
        predictions = layers.Dense(len(self.y_col), activation='sigmoid')(x)

        model = models.Model(inputs=inc_model.input, outputs=predictions)

        model.compile(
            optimizer=optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999),
            loss='binary_crossentropy',
            metrics=['accuracy', precision, recall, fscore, tf_auc])

        history = model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.train_steps_per_epoch,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=self.validation_generator,
            validation_steps=self.validation_steps_per_epoch,
            use_multiprocessing=True,
            callbacks=cb,
            class_weight=self.class_weight_list,
            verbose=False)

        model.save('logs/' + self.trainer_name + '.h5')

        return history

    def evaluating(self):
        model = models.load_model(
            'logs/' + self.trainer_name + '_best.h5',
            custom_objects={
                'precision': precision,
                'recall': recall,
                'fscore': fscore,
                'auc': tf_auc
            })

        test_result = model.evaluate_generator(
            self.test_generator,
            use_multiprocessing=True,
            steps=self.test_steps_per_epoch)
        test_result = pd.DataFrame(test_result, index=model.metrics_names).T
        test_result.to_csv(
            'logs/test_result_' + self.trainer_name + '.csv', sep='\t')

    def lr_finder(self, filename):
        losses = []
        for power in range(7):
            lr = 10**(-power - 1)
            hist = self.training(lr_search=True, lr=lr)
            losses.append([hist.history['loss'][0], power, lr])
        losses = pd.DataFrame(losses, columns=['loss', 'power', 'lr'])
        losses.to_csv('logs/lr_finder_' + filename + '.csv', sep='\t')

    def resume_training(self, pretraining_name):
        model = models.load_model(
            'logs/' + pretraining_name + '.h5',
            custom_objects={
                'precision': precision,
                'recall': recall,
                'fscore': fscore,
                'auc': tf_auc
            })

        history = model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.train_steps_per_epoch,
            epochs=self.epochs,
            initial_epoch=self.initial_epoch,
            validation_data=self.validation_generator,
            validation_steps=self.validation_steps_per_epoch,
            use_multiprocessing=True,
            callbacks=self.cb,
            class_weight=self.class_weight_list,
            verbose=False)

        model.save(self.trainer_name + '.h5')

        return history

    def auc_test(self):
        model = models.load_model(
            'logs/' + self.trainer_name + '_best.h5',
            custom_objects={
                'precision': precision,
                'recall': recall,
                'fscore': fscore,
                'auc': tf_auc
            })

        y_pred_class = model.predict_generator(
            self.test_generator, steps=self.test_steps_per_epoch)

        fpr_list = []
        tpr_list = []
        roc_auc_list = []
        for i, finding in enumerate(self.y_col):
            fpr, tpr, _ = roc_curve(
                self.test_generator.classes[:, i],
                y_pred_class[:, i],
                pos_label=1)
            roc_auc = auc(fpr, tpr)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)
        return roc_auc_list, fpr_list, tpr_list

    def Grad_CAM(self, filename):
        model = models.load_model(
            'logs/' + self.trainer_name + '_best.h5',
            custom_objects={
                'precision': precision,
                'recall': recall,
                'fscore': fscore,
                'auc': tf_auc
            })

        img_path = 'images/' + filename
        img = load_img(img_path, target_size=(224, 224))
        display(img)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 256

        preds = model.predict(x)
        x_output = model.output
        last_conv_layer = model.get_layer('bn')
        grads = K.gradients(x_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input],
                             [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])

        for i in range(len(pooled_grads_value)):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        plt.matshow(heatmap, cmap=plt.cm.magma)
        plt.savefig("logs/" + filename + "_heatmap.png")
        plt.show()

        img = cv2.imread(img_path)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        #Overlay Bounding Box
        bbox = pd.read_csv('BBox_List_2017.csv', usecols=range(6))
        bbox.columns = ['Image_Index', 'Finding_Label', 'x', 'y', 'w', 'h']
        box_list = bbox.query(
            'Image_Index == @filename and Finding_Label in @self.y_col')
        box_list = box_list[['x', 'y', 'w', 'h']].astype('int16')
        box_list = box_list.values.tolist()
        box_list = [[(x[0], x[1]), (x[0] + x[2], x[1] + x[3])]
                    for x in box_list]
        if len(box_list) > 0:
            for box in box_list:
                cv2.rectangle(
                    superimposed_img, box[0], box[1], (0, 255, 0), thickness=3)

        cv2.imwrite("logs/" + filename + "_superimposed_img.png",
                    superimposed_img)
        img = load_img(
            "logs/" + filename + "_superimposed_img.png",
            target_size=(224, 224))
        display(img)
