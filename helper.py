import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import ensemble
import keras
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.models import Sequential
import scipy.io.wavfile as wav
import pydub
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def build_mlp(x_dim, dropout=0):
    '''
    Functionality: a simple MLP predictor for music character
    
    x_dim: dimension of the feature vector
    dropout: dropout rate
    '''
    model = Sequential()
    model.add(Dense(64, input_dim=x_dim, kernel_initializer='normal', activation='relu')) 
    #128 is the number of units in the first hidden layer
    model.add(Dropout(dropout))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    # Compile model
    model.compile(loss='mse', optimizer='adam')
    
    return model


def predictor_fitted_assess_potential(X_train, y_train, X_test, y_test, method):
    '''
    Functionality: to use specified method (model) for the training data, and evaluate the trained model on test data
    X_train: training data on features [type: numpy array, shape: (number of training examples, number of features)]
    y_train: training data on label [type: numpy array, shape: (number of training examples, None)]
    X_test: test data on features [type: numpy array, shape: (number of test examples, number of features)]
    y_test: test data on label [type: numpy array, shape: (number of test examples, None)]
    method: method or model used to train the data [type: string]
    
    outputs:
    y_predicted_train: predicted labels for training data [type: numpy array, shape: (number of training examples, None)]
    y_predicted_test: predicted labels for test data [type: numpy array, shape: (number of test example, None)]
    predictor: the trained model
    r2_train: r2 score (can be negative) for the training data [type: float, shape: None]
    r2_test:  r2 score (can be negative) for the test data [type: float, shape: None]
    '''
    if method == 'Lasso':
        lasso = sk.linear_model.Lasso().fit(X_train,y_train)
        y_predicted_train = lasso.predict(X_train)
        y_predicted_test = lasso.predict(X_test)
        predictor = lasso
        
    if method == 'Ridge':
        ridge = sk.linear_model.Ridge().fit(X_train,y_train)
        y_predicted_train = ridge.predict(X_train)
        y_predicted_test = ridge.predict(X_test)
        predictor = ridge
        
    if method == 'Random Forest':
        rf = sk.ensemble.RandomForestRegressor().fit(X_train,y_train)
        y_predicted_train = rf.predict(X_train)
        y_predicted_test = rf.predict(X_test)
        predictor = rf
    
    if method == 'GBDT':
        
        gbdt = sk.ensemble.GradientBoostingRegressor(random_state=0, loss='ls').fit(X_train, y_train) 
        y_predicted_train = gbdt.predict(X_train)
        y_predicted_test = gbdt.predict(X_test)
        predictor = gbdt
    
    if method == 'MLP':
        x_dim = X_train.shape[1]
        mlp = build_mlp(x_dim, 0.5)
        mlp.fit(X_train.as_matrix(), y_train.as_matrix(), epochs=5, batch_size=100, verbose=0)
        y_predicted_train = mlp.predict(X_train.as_matrix())
        y_predicted_test = mlp.predict(X_test.as_matrix())
        predictor = mlp
                               
    mse_train = mean_squared_error(y_train.as_matrix(), y_predicted_train)
    mse_test = mean_squared_error(y_test.as_matrix(), y_predicted_test)
    r2_train = r2_score(y_train.as_matrix(), y_predicted_train)
    r2_test = r2_score(y_test.as_matrix(), y_predicted_test)
    
    #print('MSE for Training Data {0:.4f}'.format(mean_squared_error(y_train.as_matrix(), y_predicted_train)))
    #print('R2 for Training Data {0:.4f}'.format(r2_score(y_train.as_matrix(), y_predicted_train)))
    #print('MSE for Testing Data {0:.4f}'.format(mean_squared_error(y_test.as_matrix(), y_predicted_test)))
    #print('R2 for Testing Data {0:.4f}'.format(r2_score(y_test.as_matrix(), y_predicted_test)))
        
    return y_predicted_train, y_predicted_test, predictor, r2_train, r2_test


def predictor_fitted_gbdt_best(X_train, y_train, X_test, y_test):
    '''
    Functionality: to use GBDT for the training data, and evaluate the trained model on test data
    X_train: training data on features [type: numpy array, shape: (number of training examples, number of features)]
    y_train: training data on label [type: numpy array, shape: (number of training examples, None)]
    X_test: test data on features [type: numpy array, shape: (number of test examples, number of features)]
    y_test: test data on label [type: numpy array, shape: (number of test examples, None)]
    
    outputs:
    y_predicted_train: predicted labels for training data [type: numpy array, shape: (number of training examples, None)]
    y_predicted_test: predicted labels for test data [type: numpy array, shape: (number of test example, None)]
    predictor: the trained model
    r2_train: r2 score (can be negative) for the training data [type: float, shape: None]
    r2_test:  r2 score (can be negative) for the test data [type: float, shape: None]
    '''
    
    #GridSearch After Deciding to use GDBT
    param_grid = {
        'max_depth': [3,6,9]
        #'max_features': [50,100,150,200,400],
        #'min_samples_leaf': [10,30,50,100]
    }
    gbdt = sk.ensemble.GradientBoostingRegressor(random_state = 0, loss = 'ls')
    Grid_CV_gbdt = GridSearchCV(estimator = gbdt, param_grid = param_grid, cv = 5)
    Grid_CV_gbdt.fit(X_train, y_train) 
    
    y_predicted_train = Grid_CV_gbdt.best_estimator_.predict(X_train)
    y_predicted_test = Grid_CV_gbdt.best_estimator_.predict(X_test)
    predictor = Grid_CV_gbdt.best_estimator_
                               
    mse_train = mean_squared_error(y_train.as_matrix(), y_predicted_train)
    mse_test = mean_squared_error(y_test.as_matrix(), y_predicted_test)
    r2_train = r2_score(y_train.as_matrix(), y_predicted_train)
    r2_test = r2_score(y_test.as_matrix(), y_predicted_test)
        
    return y_predicted_train, y_predicted_test, predictor, r2_train, r2_test



def plot_fit(y_train, y_predicted_train, y_test, y_predicted_test, method = '(Ridge Regression)'):
    '''
    Functionality: plot fitted labels and true labels for both training data and test data under specified method, to visualize
    the fit of the predicted labels, and compare it with the random prediction.
    
    y_train: true label for the training data [type: numpy array, shape: (number of training examples, None)]
    y_predicted_train: predicted label for the training data [type: numpy array, shape: (number of training examples, None)]
    y_test: true label for the test data [type: numpy array, shape: (number of testing examples, None)]
    y_predicted_test: predicted label for the test data [type: numpy array, shape: (number of testing examples, None)]
    method: the method used to calculate the predicted label, used in the caption of graph [type: string]
    '''
    
    print('Predicted Value versus True for Training and Testing Data ' + method)
    plt.figure(figsize=(10,6))
    plt.subplot(2, 2, 1)
    plt.plot(y_train.as_matrix()[0:100],'og', label='True')
    plt.plot(y_predicted_train[0:100],'or',label='Predicted')
    plt.legend(loc='upper right')
    plt.title('Training Data'+' (GBDT)')
    plt.ylabel('y values')
    
    plt.subplot(2, 2, 2)
    plt.plot(y_test.as_matrix()[0:100],'og',label='True')
    plt.plot(y_predicted_test[0:100],'or',label='Predicted')
    plt.legend(loc='upper right')
    plt.title('Testing Data'+' (GBDT)')
    plt.ylabel('y values')
    
    #Random Prediction (uniform over [0,1])
    plt.subplot(2, 2, 3)
    plt.plot(y_train.as_matrix()[0:100],'og', label='True')
    plt.plot(np.random.uniform(0, 1, 100),'or',label='Predicted')
    plt.legend(loc='upper right')
    plt.title('Training Data'+' (Random Prediction)')
    plt.ylabel('y values')
    
    plt.subplot(2, 2, 4)
    plt.plot(y_test.as_matrix()[0:100],'og',label='True')
    plt.plot(np.random.uniform(0, 1, 100),'or',label='Predicted')
    plt.legend(loc='upper right')
    plt.title('Testing Data'+' (Random Prediction)')
    plt.ylabel('y values')

    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.show()

    
    
def find_music_indexes(char_vec, number = 10, database = 'large'):
    '''
    Functionality: find indexes of musics in the database that are closest to the specified character vector
    char_vec: a vector with each element to be one dimension to capture one aspect of the music's character, if 
    one dimension is nan, then it means that the user did not specify this dimension. This dimension will be ignored
    in finding the closest music tracks.
    [type: numpy array, shape: one dimensional array]
    
    number: number of closest musics to search
    
    Return: the indexes of the closest musics to the specified character vector, in increasing order of the distance
    [type: list, shape: of length number]
    '''
    #Create user specified dimensions to search over
    y_label = ['acousticness','danceability','energy','instrumentalness',
           'liveness','speechiness','tempo_normalized','valence'] 
    char_vec_name = []
    char_vec_value = []
    for i in range(len(char_vec)):
        if not np.isnan(char_vec[i]):
            char_vec_name.append(y_label[i])
            char_vec_value.append(char_vec[i])
    
    if database == 'large':
        data = pd.read_csv('data_full_extracted_feature_label.csv', index_col=None)
    if database == 'small':
        data = pd.read_csv('data_labeled_extracted_feature_label.csv', index_col=None)
    char_mat = data[char_vec_name].as_matrix()
    temp = np.sum((char_mat-char_vec_value)**2, axis = 1)
    indexes_tmp = temp.argsort()[:number]
    indexes = list(data['track_id'][indexes_tmp])
    
    return indexes



def wav_to_num(input_path, input_name):
    '''
    Functionality: read wav form of a music and normalize the numerical sequence to be in [-1,1]
    
    input_path: absolute path of the input music in the disk [type: string]
    input_name: name of the input music [type: string, of format 'name.mp3']
    sampling_rate: sampling rate
    output: numerical representation of wav information, normalized to be in [-1,1],
    (of size (length of wave sequence, number of channels))
    '''
    input_path_and_name = os.path.join(input_path, input_name)
    sampling_rate, data = wav.read(input_path_and_name)
    #normalize the 16-bit signal to [-1,1]
    output = data.astype('float32')/32767
    return sampling_rate, output


def num_to_fft(input_data):
    '''
    Functionality: perform fast Fourier Transformation on input_data and return frequency domain information
    
    input_data: normalized numerical representation of a music [type: numpy array, shape: (length of input_data, number of channels)]
    output: frequency domain representation of the music [type: numpy array, shape: (length of input_data*2, number of channels)]
    '''
    fft = np.fft.fft(input_data) 
    output = np.concatenate((np.real(fft), np.imag(fft)))
    return output

def num_to_fft_blockwise(input_data, block_size):
    '''
    convert from time domain numpy array to frequency domain numpy array, block by block
    
    input_data: numerical representation of wav 
    [type: numpy array, shape: (length of wave sequence, number of channels)]
    
    block_size: length of a block when we cut input_data into different blocks, each block will be of shape (block_size, number of channels). Usually block_size is set to be the sampling rate
    [type: integer, shape: 0]
    
    output: given the time steps, each row of output corresponds to one cell at the bottom of RNN (i.e. one time step)
    [type: numpy array, shape: (num_blocks, 2*block_size*num_channels)]
    '''
    num_blocks = input_data.shape[0]//block_size #without zero-padding, tail data disposed
    #num_blocks = input_data.shape[0]//block_size+1, this is the zero-padding specification
    num_channels = input_data.shape[1]
    out_shape = (num_blocks, 2*block_size, num_channels) #2 represents real and imaginary parts after fft
    out = np.zeros(out_shape)
    for i in range(num_blocks):
        #without zero-padding, tail data disposed
        tmp = np.fft.fft(input_data[i*block_size:(i+1)*block_size])
        out[i] = np.concatenate((np.real(tmp), np.imag(tmp))) # shape: (block_size*2, number of channels)
        
        #Below is zero-padding specification
        #if i < num_blocks-1:
        #    tmp = np.fft.fft(input_data[i*block_size:(i+1)*block_size])
        #    out[i] = np.concatenate((np.real(tmp), np.imag(tmp))) # shape: (block_size*2, number of channels)
        #else:
        #    tmp = np.fft.fft(input_data[i*block_size:])
        #    tail_data_dim = np.mod(input_data.shape[0],block_size)
        #    out[i][:(tail_data_dim*2)] = np.concatenate((np.real(tmp), np.imag(tmp))) 
    
    output = np.zeros((out_shape[0], out_shape[1]*num_channels))
    for j in range(num_blocks):
        output[j,:] = out.copy()[j,:,:].T.reshape(1, out_shape[1]*num_channels)
    
    return output


def build_train_data_one_song(data_fft, time_steps):
    '''
    Functionality: use FFT form of a song to build training data for this song
    Arguments:
    data_fft: FFT form of the data 
    [type: numpy array, shape: (num_blocks, 2*block_size*num_channels)]
    
    time_steps: horizontal length of the RNN
    [type: interger, shape: 0]
    
    Outputs
    X_train: input sequence traning data 
    [type: numpy array, shape: (num_examples, time_steps, input_cell_dim)]
    num_examples: number of RNN sequences obtained from one song
    input_cell_dim: dimension of each cell at one time step
    
    y_train: target sequence training data
    [type and shape: same as X_train]
    '''
    
    num_examples = data_fft.shape[0]//time_steps #By using //, the tail of the numerical representation is cut
    input_cell_dim = data_fft.shape[1]
    input_shape = (num_examples, time_steps, input_cell_dim)
    data_tmp = data_fft[:(num_examples*time_steps)]
    
    X_train = data_tmp.reshape(input_shape)
    y_train = np.zeros(X_train.shape)
    y_train[:,0,:] = X_train.copy()[:,X_train.shape[1]-1,:]
    y_train[:,1:,:] = X_train.copy()[:,:X_train.shape[1]-1,:] 
    #y_train is shifted version of X_train, the last cell is filled with the first of X_train by convention

    return X_train, y_train


def build_train_data(music_name_list, train_musics_path, block_size, time_steps):
    '''
    Functionality: build training data from musics in the neigbhorhood of specified vector of music character
    
    Arguments:
    music_name_list: list of training musics' names, each pieces is in the form 'xxx.wav'
    
    train_musics_path: path of training musics
    [type: string]
    block_size: size of blocks when we cut each track's wav into several blocks
    [type: integer, shape: 0]
    time_steps: time steps for the RNN
    [type: integer, shape: 0]
    
    return: training data for the seq2seq model. 
    X_train: input sequence 
    [type: numpy array, 
    shape: (number of examples (i.e. sequences) obtained from all the selected music tracks, time_steps, input_cell_dim)]
    y_train: output sequence 
    [type and shape: same as X_train]
    sampling_rate_vec: list of sampling rates, each element corresponds to one track
    [type: list, shape: number of music tracks for training purpose]
    '''
      
    X_train_list = []
    y_train_list = []
    sampling_rate_vec = []
    for i in music_name_list:
        sampling_rate, wav = wav_to_num(train_musics_path, i)
        sampling_rate_vec.append(sampling_rate)
        if wav.ndim == 1:
            wav_fft = num_to_fft_blockwise(wav.reshape(-1,1), block_size)
        else:
            wav_fft = num_to_fft_blockwise(wav[:,0].reshape(-1,1), block_size)
            #wav_fft = num_to_fft_blockwise(wav, block_size), this is code if we want to use all the channels
        
        X_train_one_song, y_train_one_song = build_train_data_one_song(wav_fft, time_steps)
        X_train_list.append(X_train_one_song)
        y_train_list.append(y_train_one_song)
    
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    return X_train, y_train, sampling_rate_vec


def fft_to_num(input_data):
    '''
    Functionality: converts input_data from frequency domain numpy array to time domain numpy array
    input_data: frequency domain representation of a music track [type: numpy array, shape: (length of input_data, number of channels)]
    output: time domain representation of a music track [type: numpy array, shape: (length of input_data/2, number of channels)]
    '''
    
    T = (input_data.shape[0]/2)
    real = input_data[0:T]
    imag = input_data[T:]
    tmp = real + 1.0J*imag
    output = np.fft.ifft(tmp)
    return output


def fft_to_num_blockwise(input_data, num_channels):
    '''
    Functionality: converts input_data from frequency domain numpy array to time domain numpy array, blockwise
    
    input_data: generated FFT form data 
    [type: numpy array, shape: (num_seq, time_steps, 2*block_size*num_channels)]
    
    num_channels: number of channels we used from the original training music tracks 
    [type: integer, shape: 0]
    '''
    
    block_size = int(input_data.shape[2]/2/num_channels)
    num_seq = input_data.shape[0]
    num_blocks_per_time_step = 2*num_channels
    time_steps = input_data.shape[1]
    out = []
    for i in range(num_seq):
        for j in range(time_steps):
            block = []
            for k in range(int(num_blocks_per_time_step/2)):
                tmp = input_data[i,j][2*k*block_size:(2*k+2)*block_size].reshape(2,block_size).T
                block.append((tmp[:,0]+1j*tmp[:,1]).reshape(-1,1))
            block_new = np.fft.ifft((np.concatenate(block, axis = 1)))
            out.append(np.real(block_new))
            
    output = np.concatenate(out)
    
    return output    

def num_to_wav(output_path, output_name, sampling_rate, input_data):
    '''
    Functionality: convert normalized numerical representation of a music track back to unnormalized wav form and write to disk
    
    output_path: the absolute path for the output wav [type: string]
    output_name: name of the output wav [type: string, in the form of 'xxx.wav']
    sampling_rate: sampling rate
    input_data: normalized numerical representation of the music track [type: numpy array, shape: (length of input_data, number of channels)]
    '''
       
    path_and_name = os.path.join(output_path, output_name)
    tmp = (np.real(input_data) * 32767.0).astype('int16')
    wav.write(path_and_name, sampling_rate, tmp)
    
    return

def mp3_to_wav(input_path, input_name, output_path, output_name):
    '''
    input_path: absolute path of the input music in the disk (type: string)
    input_name: name of the input music (type: string, of format 'name.mp3')
    output_path: absolute path of the output music in the disk (type: string)
    output_name: name of the output music (type: string, of format 'name.wav')
    '''
    pydub.AudioSegment.ffmpeg = 'C:/ffmpeg'
    input_path_and_name = os.path.join(input_path, input_name)
    output_path_and_name = os.path.join(output_path, output_name)
    
    tmp = pydub.AudioSegment.from_mp3(input_path_and_name)
    tmp.export(output_path_and_name, format='wav')
               
    return

def wav_to_map3(input_path, input_name, output_path, output_name):
    '''
    Functionality: convert wav form of a music track into mp3 form in order to play within python
    
    input_path: absolute path for the input wav form data [type: string]
    input_name: name of the input wav data [type: string, in the form of 'xxx.wav']
    output_path: absolute path for the output mp3 form data [type: string]
    output_name: name of the output mp3 data [type: string, in the form of 'xxx.mp3']
    '''
    
    pydub.AudioSegment.ffmpeg = 'C:/ffmpeg'
    input_path_and_name = os.path.join(input_path, input_name)
    output_path_and_name = os.path.join(output_path, output_name)
    
    sound = pydub.AudioSegment.from_wav(input_path_and_name)
    sound.export(output_path_and_name, format='mp3')
    
    return


def generate_new_music(prime_sequence, model, 
                       output_path, output_name_wav, output_name_mp3,
                       block_size, num_seq = 5, 
                       method = 'iterative'):
    '''
    Functionality: use a prime sequence to generate new music
    
    Arguments:
    prime_sequence: the FFT form the nearest music to the specified vector of music character
    [type: numpy array, shape: (num_examples in the nearest music, time_steps, input_cell_dim)]
    
    model: the trained model for music prediction from the RNN
    output_path: path to write the generated music [type: string]
    output_name_wav: name of the generated music [type: string, in the form of 'xxx.wav']
    output_name_mp3: name of the generated music [type: string, in the form of 'xxx.mp3']
    
    block_size: size of blocks when we cut each track's wav into several blocks
    [type: integer, shape: 0]
    
    num_seq: number of sequences to generate and pieced together to form the new music, only for the 'iterative' method
    
    method: either 'iterative' or 'non-iterative'
    'iterative': prime sequence is the first training sequence from the first training music track (i.e. the nearest 
    music to the specified vector of music character)
    'non-iterative': each sequence from the first training music track is used to generate a new sequence, and then all
    the new sequences are pieced together to create a new music
    
    Return:
    output: new music in mp3 format, written to specified path
    '''
    if method == 'iterative':
        seed_seq = prime_sequence[0]
        seed_seq = np.reshape(seed_seq, (1, seed_seq.shape[0], seed_seq.shape[1]))
        fft_gen = np.zeros((num_seq, seed_seq.shape[1], seed_seq.shape[2]))
        for i in range(num_seq):
            tmp = model.predict(seed_seq)
            fft_gen[i] = tmp
            seed_seq = tmp
    
    if method == 'non-iterative':
        fft_gen = np.zeros(prime_sequence.shape)
        for i in range(prime_sequence.shape[0]):
            tmp = np.reshape(prime_sequence[i], (1, prime_sequence[i].shape[0], prime_sequence[i].shape[1]))
            fft_gen[i] = model.predict(tmp)
            
    out_num = fft_to_num_blockwise(fft_gen, num_channels = 1)    
    
    num_to_wav(output_path, output_name_wav, block_size, out_num)
    wav_to_map3(output_path, output_name_wav, output_path, output_name_mp3)
    
    return

##Combine some functions above 
def wav_to_fft(input_data):
    #input_data: string, the absolute path and name of the input wav file
    sampling_rate, data = wav.read(input_data)
    #normalize the 16-bit signal to [-1,1]
    numeric = data.astype('float32')/32767 #numerica of shape Tx2
    fft = np.fft.fft(numeric) #fft of shape Tx2 
    output = np.concatenate((np.real(fft), np.imag(fft))) #output of shape 2Tx2
    
    return sampling_rate, output
    

def fft_to_mp3(data_fft, wav_path, wav_name, mp3_path, mp3_name, sampling_rate):
    
    tmp1 = fft_to_num(data_fft)
    num_to_wav(wav_path, wav_name, sampling_rate, data_fft)
    wav_to_map3(wav_path, wav_name, mp3_path, mp3_name)
    
    return
