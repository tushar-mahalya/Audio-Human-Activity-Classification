from sklearn.preprocessing import LabelEncoder, scale, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, UpSampling2D, Input
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.regularizers import l1
from keras.utils.vis_utils import plot_model
from datetime import datetime
from sklearn import metrics
import librosa, librosa.display, os, csv
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
plt.switch_backend('agg')
import itertools
import scipy as sp
import os

class AudioClassifier():
    def __init__(self):
        self.PLOT_MFCC = False
        self.target_names = ['door', 'light', 'plate']
        self.dataset = './train_test_filtered.csv'
        self.process_dataset()

    def extract_features(self, file_name):
        try:
            """
                Load and preprocess the audio
            """
            audio, sample_rate = librosa.load(file_name)
            y = audio

            """
                Convert to MFCC numpy array
            """
            max_pad_length = 431
            n_mfcc = 120
            n_fft = 4096
            hop_length = 512
            n_mels = 512
            #mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
            mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            pad_width = max_pad_length-mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
            #print(mfccs.shape)
            #mfccsscaled = np.mean(mfccs.T,axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", e)
            return None, 0

        return mfccs, sample_rate

    def process_dataset(self):
        features = []
        with open(self.dataset) as dataset:
            csv_reader = csv.reader(dataset, delimiter=',')
            index = 1

            for row in csv_reader:
                print(row)
                file_name = os.getcwd()+'/epic_audio/'+row[7]
                if not os.path.exists(file_name):
                    print(file_name, 'not found')
                    continue

                class_label = row[6]
                data, sr = self.extract_features(file_name)
                #print(data)
                if data is not None:
                    features.append([data, class_label])

                    # Save an image of the MFCC
                    if self.PLOT_MFCC:
                        self.plot_mfcc(row[7]+'_'+class_label, data, sr)
                else:
                    print("Data is empty: ", file_name)

                print("Processed row ", index)
                index = index+1

        # Convert into a Panda dataframe
        featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
        print(featuresdf)

        print('Finished feature extraction from ', len(featuresdf), ' files')

        # Convert features and corresponding classification labels into numpy arrays
        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.class_label.tolist())

        # Encode the classification labels
        le = LabelEncoder()
        yy = to_categorical(le.fit_transform(y))

        # Train the model and save the results
        self.CNN(X, yy)

    def CNN(self, X, yy):
        ### CNN

        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

        # Reshape the data
        num_rows = 120
        #num_rows = 12
        num_columns = 431
        #num_columns = 120
        num_channels = 1
        num_labels = yy.shape[1]

        x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
        x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        # Construct model
        model = Sequential()

        model.add(Conv2D(16, (7,7), input_shape=(num_rows, num_columns, num_channels), activation='relu', padding="same"))
        model.add(BatchNormalization())
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.2))

        model.add(Conv2D(32, (3,3), activation='relu', padding="same"))
        model.add(BatchNormalization())
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.2))

        model.add(Conv2D(64, (3,3), activation='relu', padding="same"))
        model.add(BatchNormalization())
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.2))

        model.add(Conv2D(128, (3,3), activation='relu', padding="same"))
        model.add(BatchNormalization())
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.2))

        model.add(Conv2D(256, (3,3), activation='relu', padding="same"))
        model.add(BatchNormalization())
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.2))

        model.add(Conv2D(512, (1,1), activation='relu', padding="same"))
        model.add(BatchNormalization())
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.2))

        #model.add(Conv2D(1024, (1,1), activation='relu', padding="same"))
        #model.add(BatchNormalization())
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.2))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(num_labels, activation='softmax'))

        learning_rate = 0.00001
        #opt = optimizers.SGD(lr=learning_rate, nesterov=True)
        opt = optimizers.Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
        model.summary()

        # Calculate pre-training accuracy
        score = model.evaluate(x_test, y_test, verbose=1)
        accuracy = 100*score[1]
        print("Pre-training accuracy: %.4f%%" % accuracy)

        # Train the model
        num_epochs = 1000
        num_batch_size = 10
        start = datetime.now()

        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', verbose=1, save_best_only=True)
        es_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.0001, patience=7, verbose=1, mode='auto', min_delta=0.001, cooldown=1, min_lr=0)
        history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_split=0.2, shuffle=False, callbacks = [checkpointer, es_callback], verbose=2)
        #history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[checkpointer], verbose=1)
        duration = datetime.now() - start
        print("Training completed in time: ", duration)

        # Evaluating the model on the training and testing set
        score = model.evaluate(x_train, y_train, verbose=0)
        print("Training Accuracy: ", score[1])

        score = model.evaluate(x_test, y_test, verbose=0)
        print("Testing Accuracy: ", score[1])

        # Plots and reports
        self.plot_graphs(history)

        y_pred = model.predict(x_train, batch_size=15)
        cm = confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=1))
        self.plot_confusion_matrix(cm, self.target_names)

        self.plot_classification_report(y_train.argmax(axis=1), y_pred.argmax(axis=1))

        plot_model(model, to_file='graphs/model_plot.png', show_shapes=True, show_layer_names=True)

        print('Complete.')

    def plot_graphs(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()
        plt.savefig('graphs/accuracy.png')
        plt.clf()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()
        plt.savefig('graphs/loss.png')
        plt.close()

    def plot_classification_report(self, x_test, y_test):
        # Print
        print(classification_report(x_test, y_test, target_names=self.target_names))
        # Save data
        clsf_report = pd.DataFrame(classification_report(y_true = x_test, y_pred = y_test, output_dict=True, target_names=self.target_names)).transpose()
        clsf_report.to_csv('graphs/classification_report.csv', index= True)

    def plot_confusion_matrix(self, cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
        matplotlib.rcParams.update({'font.size': 22})
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(14, 12))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig('graphs/confusion_matrix.png', bbox_inches = "tight")
        plt.close()

    def plot_mfcc(self, filename, mfcc, sr):
        plt.figure(figsize=(10, 4))
        #S_dB = librosa.power_to_db(mfcc, ref=np.max)
        #librosa.display.specshow(S_dB, y_axis='mel', x_axis='time')
        librosa.display.specshow(librosa.amplitude_to_db(mfcc, ref=np.max), y_axis='mel', x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title(filename)
        plt.tight_layout()
        pylab.savefig('mfcc/'+filename+'.png', bbox_inches=None, pad_inches=0)
        pylab.close()

AudioClassifier()
