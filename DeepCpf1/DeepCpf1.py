import sys

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv1D, AveragePooling1D
from keras.layers import Multiply
from numpy import zeros


def main():
    Seq_deepCpf1_Input_SEQ = Input(shape=(34, 4))
    Seq_deepCpf1_C1 = Conv1D(80, 5, activation='relu')(Seq_deepCpf1_Input_SEQ)
    Seq_deepCpf1_P1 = AveragePooling1D(2)(Seq_deepCpf1_C1)
    Seq_deepCpf1_F = Flatten()(Seq_deepCpf1_P1)
    Seq_deepCpf1_DO1= Dropout(0.3)(Seq_deepCpf1_F)
    Seq_deepCpf1_D1 = Dense(80, activation='relu')(Seq_deepCpf1_DO1)
    Seq_deepCpf1_DO2= Dropout(0.3)(Seq_deepCpf1_D1)
    Seq_deepCpf1_D2 = Dense(40, activation='relu')(Seq_deepCpf1_DO2)
    Seq_deepCpf1_DO3= Dropout(0.3)(Seq_deepCpf1_D2)
    Seq_deepCpf1_D3 = Dense(40, activation='relu')(Seq_deepCpf1_DO3)
    Seq_deepCpf1_DO4= Dropout(0.3)(Seq_deepCpf1_D3)
    Seq_deepCpf1_Output = Dense(1, activation='linear')(Seq_deepCpf1_DO4)
    Seq_deepCpf1 = Model(inputs=[Seq_deepCpf1_Input_SEQ], outputs=[Seq_deepCpf1_Output])

    Seq_deepCpf1.load_weights('weights/Seq_deepCpf1_weights.h5', by_name=True)

    FILE = open(sys.argv[1], "r")
    data = FILE.readlines()
    SEQ, CA = PREPROCESS(data)
    FILE.close()
    
    Seq_deepCpf1_SCORE = Seq_deepCpf1.predict([SEQ], batch_size=50, verbose=0)

    OUTPUT = open(sys.argv[2], "w")
    for l in range(len(data)):
        if l == 0:
            OUTPUT.write(data[l].strip())
            OUTPUT.write("\tSeq-deepCpf1 Score\n")
        else:
            OUTPUT.write(data[l].strip())
            OUTPUT.write("\t%f\n" % (Seq_deepCpf1_SCORE[l-1]))
    OUTPUT.close()
    
def PREPROCESS(lines):
    data_n = len(lines) - 1
    SEQ = zeros((data_n, 34, 4), dtype=int)
    CA = zeros((data_n, 1), dtype=int)
    
    for l in range(1, data_n+1):
        data = lines[l].split()
        seq = data[1]
        for i in range(34):
            if seq[i] in "Aa":
                SEQ[l-1, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l-1, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l-1, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l-1, i, 3] = 1
        CA[l-1,0] = int(data[2])*100

    return SEQ, CA


if __name__ == '__main__':
        main()
        
