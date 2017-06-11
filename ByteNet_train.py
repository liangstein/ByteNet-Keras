import os;
import numpy as np;
from keras.models import Model;
from keras.layers.embeddings import Embedding;
from keras.models import Sequential,load_model;
from keras.optimizers import rmsprop,adam,adagrad,SGD;
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau;
from keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer;
from keras.layers import Input,Dense,merge,Dropout,BatchNormalization,Activation,Conv1D;

# setting current working directory
WKDIR=os.getcwd();

def load_dataset(batch_size,N=150000):
    French = list(np.load(WKDIR + "/french_sentences.npy")[:N]);# read dataset
    English = list(np.load(WKDIR + "/english_sentences.npy")[:N]);
    English = [i + "\n" for i in English];# add ending signal at the sequence end
    while 1:
        if len(English) % batch_size != 0:
            del English[-1];
            del French[-1];
        else:
            break;
    return French,English;

def build_vacabulary(French,English):
    all_eng_words = [];
    all_french_words = [];
    for i in np.arange(0, len(French)):
        all_eng_words.append(English[i]);
        all_french_words.append(French[i]);
    tokeng = Tokenizer(char_level=True);
    tokeng.fit_on_texts(all_eng_words);
    eng_index = tokeng.word_index;  # build character to index dictionary
    index_eng = dict((eng_index[i], i) for i in eng_index);
    tokita = Tokenizer(char_level=True);
    tokita.fit_on_texts(all_french_words);
    french_index = tokita.word_index;  # build character to index dictionary
    index_french = dict((french_index[i], i) for i in french_index);
    return (eng_index,french_index,index_eng,index_french);

# convert a batch of input sequences to tensors
def generate_batch_data(English,French,eng_index,french_index,batch_size):
    while 1:
        all_labels=np.arange(0,len(French));np.random.shuffle(all_labels);
        batch_labels=np.array_split(all_labels,int(len(French)*batch_size**-1));
        for labels in batch_labels:
            source_vec=np.zeros((batch_size,maxlen+1),dtype=np.uint16);
            target0_vec=np.zeros((batch_size,maxlen),dtype=np.uint16);
            target1_vec = np.zeros((batch_size, maxlen+1, len(eng_index)), dtype=np.uint16);
            sampleweights=np.zeros((batch_size,maxlen+1),dtype=np.uint16);
            for i,a in enumerate(labels):
                for j1,ele1 in enumerate(French[a]):
                    source_vec[i,j1]=french_index[ele1];
                for j2,ele2 in enumerate(English[a][:-1]):
                    target0_vec[i,j2]=eng_index[ele2];
                for j3,ele3 in enumerate(English[a]):
                    target1_vec[i,j3,eng_index[ele3]-1]=1;
                    sampleweights[i,j3]=1;# mask the loss function
            t0=np.zeros((batch_size,1,500),dtype=np.uint8);# beginning of target sequence
            yield ([source_vec,target0_vec,t0],target1_vec,sampleweights);

def build_model(french_index,eng_index,index_french,index_eng,English,French):
    input_sequence = Input(shape=(maxlen + 1,));
    input_tensor = Embedding(input_length=maxlen + 1, input_dim=len(french_index) + 1, output_dim=500)(input_sequence);
    encoder1 = Conv1D(filters=500, kernel_size=1, strides=1, padding="same")(input_tensor);
    encoder1 = Activation("relu")(encoder1);
    encoder1 = Conv1D(filters=250, kernel_size=5, strides=1, padding="same", dilation_rate=1)(encoder1);
    encoder1 = BatchNormalization(axis=-1)(encoder1);
    encoder1 = Activation("relu")(encoder1);
    encoder1 = Conv1D(filters=500, kernel_size=1, strides=1, padding="same")(encoder1);
    input_tensor = merge([input_tensor, encoder1], mode="sum");
    encoder2 = BatchNormalization(axis=-1)(input_tensor);
    encoder2 = Activation("relu")(encoder2);
    encoder2 = Conv1D(filters=500, kernel_size=1, strides=1, padding="same")(input_tensor);
    encoder2 = BatchNormalization(axis=-1)(encoder2);
    encoder2 = Activation("relu")(encoder2);
    encoder2 = Conv1D(filters=250, kernel_size=5, strides=1, padding="same", dilation_rate=2)(encoder2);
    encoder2 = BatchNormalization(axis=-1)(encoder2);
    encoder2 = Activation("relu")(encoder2);
    encoder2 = Conv1D(filters=500, kernel_size=1, strides=1, padding="same")(encoder2);
    input_tensor = merge([input_tensor, encoder2], mode="sum");
    encoder3 = BatchNormalization(axis=-1)(input_tensor);
    encoder3 = Activation("relu")(encoder3);
    encoder3 = Conv1D(filters=500, kernel_size=1, strides=1, padding="same")(encoder3);
    encoder3 = BatchNormalization(axis=-1)(encoder3);
    encoder3 = Activation("relu")(encoder3);
    encoder3 = Conv1D(filters=250, kernel_size=5, strides=1, padding="same", dilation_rate=4)(encoder3);
    encoder3 = BatchNormalization(axis=-1)(encoder3);
    encoder3 = Activation("relu")(encoder3);
    encoder3 = Conv1D(filters=500, kernel_size=1, strides=1, padding="same")(encoder3);
    input_tensor = merge([input_tensor, encoder3], mode="sum");
    encoder4 = BatchNormalization(axis=-1)(input_tensor);
    encoder4 = Activation("relu")(encoder4);
    encoder4 = Conv1D(filters=500, kernel_size=1, strides=1, padding="same")(encoder4);
    encoder4 = BatchNormalization(axis=-1)(encoder4);
    encoder4 = Activation("relu")(encoder4);
    encoder4 = Conv1D(filters=250, kernel_size=5, strides=1, padding="same", dilation_rate=8)(encoder4);
    encoder4 = BatchNormalization(axis=-1)(encoder4);
    encoder4 = Activation("relu")(encoder4);
    encoder4 = Conv1D(filters=500, kernel_size=1, strides=1, padding="same")(encoder4);
    input_tensor = merge([input_tensor, encoder4], mode="sum");
    encoder5 = BatchNormalization(axis=-1)(input_tensor);
    encoder5 = Activation("relu")(encoder5);
    encoder5 = Conv1D(filters=500, kernel_size=1, strides=1, padding="same")(encoder5);
    encoder5 = BatchNormalization(axis=-1)(encoder5);
    encoder5 = Activation("relu")(encoder5);
    encoder5 = Conv1D(filters=250, kernel_size=5, strides=1, padding="same", dilation_rate=16)(encoder5);
    encoder5 = BatchNormalization(axis=-1)(encoder5);
    encoder5 = Activation("relu")(encoder5);
    encoder5 = Conv1D(filters=500, kernel_size=1, strides=1, padding="same")(encoder5);
    input_tensor = merge([input_tensor, encoder5], mode="sum");
    input_tensor = Activation("relu")(input_tensor);
    input_tensor = Conv1D(filters=500, kernel_size=1, padding="same", activation="relu")(input_tensor);
    target_sequence = Input(shape=(maxlen,));
    t0 = Input(shape=(1, 500));
    target_input = Embedding(input_length=maxlen, input_dim=len(eng_index) + 1, output_dim=500)(target_sequence);
    target_input = merge([t0, target_input], concat_axis=1, mode="concat");
    input_to_decoder_sequence = merge([input_tensor, target_input], concat_axis=-1, mode="concat");
    decoder1 = Conv1D(filters=1000, kernel_size=1, padding="same")(input_to_decoder_sequence);
    decoder1 = BatchNormalization(axis=-1)(decoder1);
    decoder1 = Activation("relu")(decoder1);
    decoder1 = Conv1D(filters=500, kernel_size=3, padding="causal", dilation_rate=1)(decoder1);
    decoder1 = BatchNormalization(axis=-1)(decoder1);
    decoder1 = Activation("relu")(decoder1);
    decoder1 = Conv1D(filters=1000, kernel_size=1, padding="same")(decoder1);
    output_tensor = merge([input_to_decoder_sequence, decoder1], mode="sum");
    decoder2 = BatchNormalization(axis=-1)(output_tensor);
    decoder2 = Activation("relu")(decoder2);
    decoder2 = Conv1D(filters=1000, kernel_size=1, strides=1, padding="same")(decoder2);
    decoder2 = BatchNormalization(axis=-1)(decoder2);
    decoder2 = Activation("relu")(decoder2);
    decoder2 = Conv1D(filters=500, kernel_size=3, padding="causal", dilation_rate=2)(decoder2);
    decoder2 = BatchNormalization(axis=-1)(decoder2);
    decoder2 = Activation("relu")(decoder2);
    decoder2 = Conv1D(filters=1000, kernel_size=1, padding="same")(decoder2);
    output_tensor = merge([output_tensor, decoder2], mode="sum");
    decoder3 = BatchNormalization(axis=-1)(output_tensor);
    decoder3 = Activation("relu")(decoder3);
    decoder3 = Conv1D(filters=1000, kernel_size=1, strides=1, padding="same")(decoder3);
    decoder3 = BatchNormalization(axis=-1)(decoder3);
    decoder3 = Activation("relu")(decoder3);
    decoder3 = Conv1D(filters=500, kernel_size=3, padding="causal", dilation_rate=4)(decoder3);
    decoder3 = BatchNormalization(axis=-1)(decoder3);
    decoder3 = Activation("relu")(decoder3);
    decoder3 = Conv1D(filters=1000, kernel_size=1, padding="same")(decoder3);
    output_tensor = merge([output_tensor, decoder3], mode="sum");
    decoder4 = BatchNormalization(axis=-1)(output_tensor);
    decoder4 = Activation("relu")(decoder4);
    decoder4 = Conv1D(filters=1000, kernel_size=1, strides=1, padding="same")(decoder4);
    decoder4 = BatchNormalization(axis=-1)(decoder4);
    decoder4 = Activation("relu")(decoder4);
    decoder4 = Conv1D(filters=500, kernel_size=3, padding="causal", dilation_rate=8)(decoder4);
    decoder4 = BatchNormalization(axis=-1)(decoder4);
    decoder4 = Activation("relu")(decoder4);
    decoder4 = Conv1D(filters=1000, kernel_size=1, padding="same")(decoder4);
    output_tensor = merge([output_tensor, decoder4], mode="sum");
    decoder5 = BatchNormalization(axis=-1)(output_tensor);
    decoder5 = Activation("relu")(decoder5);
    decoder5 = Conv1D(filters=1000, kernel_size=1, strides=1, padding="same")(decoder5);
    decoder5 = BatchNormalization(axis=-1)(decoder5);
    decoder5 = Activation("relu")(decoder5);
    decoder5 = Conv1D(filters=500, kernel_size=3, padding="causal", dilation_rate=16)(decoder5);
    decoder5 = BatchNormalization(axis=-1)(decoder5);
    decoder5 = Activation("relu")(decoder5);
    decoder5 = Conv1D(filters=1000, kernel_size=1, padding="same")(decoder5);
    output_tensor = merge([output_tensor, decoder5], mode="sum");
    output_tensor = Activation("relu")(output_tensor);
    # decoder=Dropout(0.1)(decoder);
    result = Conv1D(filters=len(eng_index), kernel_size=1, padding="same", activation="softmax")(output_tensor);
    model = Model(inputs=[input_sequence, target_sequence, t0], outputs=result);
    opt = adam(lr=0.0003); # as in the paper, we choose adam optimizer with lr=0.0003
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'],
                  sample_weight_mode="temporal");
    return model;

def train(batch_size,epochs,maxlen):
    French,English=load_dataset(batch_size);
    eng_index, french_index, index_eng, index_french=build_vacabulary(French,English);
    model=build_model(french_index,eng_index,index_french,index_eng,English,French);
    early = EarlyStopping(monitor="loss", mode="min", patience=10);
    lr_change = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=0, min_lr=0.000)
    checkpoint = ModelCheckpoint(filepath=WKDIR + "/conv1d_french_eng",
                                 save_best_only=False);# checkpoint the model after each epoch
    # start training !
    model.fit_generator(generate_batch_data(English,French,eng_index,french_index,batch_size),
                        steps_per_epoch=int(len(English) * batch_size ** -1),
                        nb_epoch=epochs, workers=1, callbacks=[early, checkpoint, lr_change], initial_epoch=0);
    model.save(WKDIR + "/conv1d_french_eng.h5")# where the model is saved

if __name__=="__main__":
    batch_size = 50;
    maxlen = 201;
    epochs=1000
    train(batch_size,epochs,maxlen);# run baby run !