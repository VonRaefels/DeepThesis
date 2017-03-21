import gammatone.gtgram
import scipy.io.wavfile
from python_speech_features import mfcc
from python_speech_features import delta
import numpy as np
from scipy.io.wavfile import write


main_dir = '/home/jorge/Documents/data/'
arma = True
LC = -8
wlen = 0.02
wstep = 0.01
channels = 64
stack = 1

# Not circular for the moment, just take the last n equally
def stack_feats(n, mat):
    nr = mat.shape[0]
    nc = mat.shape[1]
    big_mat = np.zeros((nr*n, nc), float)
    aux = mat.T
    idx = 0
    for row in aux:
        for i in range(0, n):
            j = i + idx
            if j >= nc:
                j = nc - 1
            big_mat[i*nr:i*nr + nr, idx] = aux[j,:]
        idx = idx + 1
    return big_mat

def extract_feat(index, name):
    rate, data = scipy.io.wavfile.read(main_dir + index + '/' + name)
    gtgram = gammatone.gtgram.gtgram(data, rate, wlen, wstep, channels, 20)
    mfcc_feat = mfcc(data ,samplerate=16000, winlen=wlen, winstep=wstep, numcep=13, nfilt=26,nfft=512,lowfreq=20,highfreq=None,preemph=0.97, ceplifter=22,appendEnergy=True)
    d_mfcc_feat = delta(mfcc_feat, 2)
    d_gtgram = delta(np.transpose(gtgram), 2)
    feat_vec = np.transpose(np.concatenate((mfcc_feat, d_mfcc_feat, np.transpose(gtgram), d_gtgram), 1))
    arma_feat_vec = np.zeros(feat_vec.shape, float)
    ret = { 'gtgram': gtgram, 'arma': feat_vec}
    if arma:
        aux = 0
        for row in feat_vec.T:
            aux_vec = np.array([])
            if aux == 0:
                aux_vec = (np.add(np.add(row, feat_vec[:, aux + 1]), feat_vec[:,aux + 2]) / 3)
            elif aux == 1:
                aux_vec = (np.add(np.add(np.add(arma_feat_vec[:,aux - 1], row), feat_vec[:,aux + 1]), feat_vec[:,aux + 2]) / 4)
            elif aux == len(feat_vec.T) - 2:
                aux_vec = (np.add(np.add(np.add(arma_feat_vec[:,aux - 2], arma_feat_vec[:,aux - 1]), row), feat_vec[:,aux + 1]) / 4)
            elif aux == len(feat_vec.T) - 1:
                aux_vec = (np.add(np.add(arma_feat_vec[:,aux - 2], arma_feat_vec[:,aux - 1]), row) / 3)
            else:
                aux_vec = (np.add(np.add(np.add(np.add(arma_feat_vec[:,aux - 2], arma_feat_vec[:,aux - 1]), row), feat_vec[:,aux + 1]), feat_vec[:,aux + 2]) / 5)
            arma_feat_vec[:,aux] = np.copy(aux_vec)
            aux = aux + 1

        ret = {'gtgram': gtgram, 'arma': arma_feat_vec}

    if stack > 1:
        vec = stack_feats(stack, arma_feat_vec)
        ret = {'gtgram': gtgram, 'arma': vec}
    return ret


def ibm_map(snr):
    aux = 0
    for element in snr:
        if element > LC:
            snr[aux] = 1
        else:
            snr[aux] = 0
        aux = aux + 1
    return snr

def add_min(row):
    aux = 0
    for val in row:
        if val == 0:
            row[aux] = 0.000000000000000000001
        aux = aux + 1
    return row

def IBM(target, echo):
    aux = 0
    echo['gtgram'] = np.apply_along_axis(add_min, 0, echo['gtgram'])
    ratio = (np.square(target['gtgram']) / np.square(echo['gtgram']))
    ratio = np.apply_along_axis(add_min, 0, ratio)
    snr = 10*np.log10(ratio)
    return np.ndarray.astype(np.apply_along_axis(ibm_map, 0, snr), int)


import sys
import time
# 0. Number of directories
# 1. main dir to scan
# 2. arma 0/1
# 3. LC in db
# 4. out input name
# 5. out target name
# 6. stack
# python3 feat_extract.py 5 /home/jorge/Documents/data/ 1 -8 dense1 dense1 2
if __name__ == "__main__":
    print('Starting feature extraction...')
    np.set_printoptions(threshold=np.inf)
    args = sys.argv
    N = int(args[1])
    main_dir = args[2]
    arma = (args[3] == '1')
    LC = int(args[4])
    out_input = args[5]
    out_targets = args[6]
    stack = int(args[7])

    input_mat = np.array([])
    target_mat = np.array([])
    for i in range(1, N + 1):
        print(i)
        mix1 = extract_feat(str(i), 'mix1.wav')
        mix2 = extract_feat(str(i), 'mix2.wav')
        mic1_target = extract_feat(str(i), 'mic1_target.wav')
        mic1_echo = extract_feat(str(i), 'mic1_echo.wav')
        target = extract_feat(str(i), 'target.wav')
        echo = extract_feat(str(i), 'echo.wav')

        ibm = IBM(mic1_target, mic1_echo)
        #print(mix1['arma'].shape)
        #print(ibm.shape)
        print('---------------------')
        #print(ibm)
        input_vec = np.concatenate((mix1['arma'], mix2['arma']), 0)
        if input_mat.size == 0:
            input_mat = np.copy(input_vec)
            target_mat = np.copy(ibm)
        else:
            input_mat = np.concatenate((input_mat, input_vec), 1)
            target_mat = np.concatenate((target_mat, ibm), 1)


    data = {'input': input_mat, 'targets': target_mat}
    #TODO put in name date, using AR model...etc....
    timestr = time.strftime("%Y%m%d-%H%M%S")
    np.save('./feats/inputs_' + out_input + '_' + timestr + '.npy', input_mat, allow_pickle=True)
    np.save('./feats/targets_' + out_targets + '_' + timestr + '.npy', target_mat, allow_pickle=True)
    print('finished')
    print(input_mat.shape)
    print(target_mat.shape)