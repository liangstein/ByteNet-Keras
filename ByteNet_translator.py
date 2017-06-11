import os;
import pickle;
import numpy as np;
from keras.preprocessing.text import text_to_word_sequence;
from keras.models import Sequential,load_model;
from keras.utils.vis_utils import plot_model;
maxlen=201;DIR=os.getcwd();
with open(DIR+"/french_index","rb") as f:
    french_index=pickle.load(f);

with open(DIR+"/eng_index","rb") as f:
    eng_index=pickle.load(f);

with open(DIR+"/index_french","rb") as f:
    index_french=pickle.load(f);

with open(DIR+"/index_eng","rb") as f:
    index_eng=pickle.load(f);

model=load_model(DIR+"/conv1d_french_eng");
def input_2_vec(input):
    source=input[0];target=input[1];
    source_vec=np.zeros((1,maxlen+1),dtype=np.uint16);
    target_vec=np.zeros((1,maxlen),dtype=np.uint16);
    for i,ele1 in enumerate(source):
        source_vec[0,i]=french_index[ele1];
    for j,ele2 in enumerate(target):
        target_vec[0,j]=eng_index[ele2];
    return (source_vec,target_vec);

def T(sentence):
    vec=input_2_vec(sentence);
    source_vec=vec[0];target_vec=vec[1];
    t0=np.zeros((1,1,500),dtype=np.uint8);
    predict=model.predict([source_vec,target_vec,t0])[0];
    predict_max=np.argmax(predict,axis=-1);
    answers=[index_eng[j+1] for j in predict_max];
    a="".join(answers);
    return a;

#plot_model(model, to_file="model.png", show_shapes=True)
WKDIR=os.getcwd();
def translate(french_sentence):
    process_english_sentence="";
    length=0;
    while 1:
        predicted_english_sentence=T([french_sentence,process_english_sentence]);
        process_english_sentence=predicted_english_sentence[:length+1];length+=1;
        if process_english_sentence[-1]=="\n":break;
        if length>=maxlen+1:break;
        #if length%10==0:print("{} completed".format(str(length*maxlen**-1)));
    return process_english_sentence;
