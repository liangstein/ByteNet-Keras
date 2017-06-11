## ByteNet-Keras
French to English translator on character level implemented by Keras

## Dependency
* Python3.6(numpy, scipy, pickle, h5py),
* Keras2.02,
* Tensorflow v1.1 backend, (Not tested with Theano backend)
* Cuda8.0, Cudnn6.0 (If GPU is used)

## Neural Network Implementation
ByteNet is a character level translation model designed by DeepMind. It was firstly raised in the paper [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099). The architecture is the neural network is :
<p align="center">
  <img src="https://github.com/liangstein/ByteNet-Keras/blob/master/structure.png" width="400"/>
</p>

## Dataset
The dataset used is [European Parliament Proceedings Parallel Corpus v7](http://www.statmt.org/europarl/). To avoid too much training time, only 150000 sentence pairs were used. The training time on a GTX 1080 is 45 hours. After 144 epochs, the categorical crossentropy is reduced to 0.001

## Models
The main difference between the models in the paper are:
1. I've used BatchNormalization instead of LayerNormalization in the paper
2. The dimension of the network is smaller due to a smaller dataset. The latent dimension is 500.

The structure of the model is:
<p align="center">
  <img src="https://github.com/liangstein/ByteNet-Keras/blob/master/model.png" height="2048"/>
</p>

## Translation effects
The comparison between Google translation is listed, right now the model is good at translating short sentences. Perhaps a max-length of 200 is too long, and training the sequences on batches based on varying lengths is necessary. 

|English Sentence|Translated Sentence|Google Translated|
|----------------|-------------------|-----------------|
|Merci beaucoup, Madame Thyssen.|Thank you very much, Mrs Haug.|Thank you very much, Ms. Thyssen.|
|Nous ferons les vérifications qui s'imposent, car, bien entendu, le procès-verbal a été adopté ; par conséquent, il faudra apporter une correction technique dans votre cas.|We will see what nevertheless encourage, and so name against the Russian report, it has taken out objective a credible and receive more to a cautious level.|We will do the necessary checks, because, of course, the minutes have been adopted; Therefore, you will need to make a technical correction in your case.|
|On y cite Mme Reding.|On my colleague, Mrs González Álvarez on I did.|On y cite Me Reading.|
|Pouvez-vous, s'il vous plaît, rectifier cela ?|Could you tell us the real declaration?|Could you please rectify that?|
|Madame Lulling, je ne peux pas rectifier parce que vous ne faites pas l' objet de ce rapport.|Mrs Lulling, I cannot see any reason to tell the last of that office this report.|Mrs. Lulling, I can not rectify because you are not the subject of this report.|
|Vous avez été élue le 16 septembre - comme vous le dites parfaitement - et ce rapport concerne les élus du 13 juin.|In my visit we have found Antonio German, you are only agreed, and we will have narrow the issues about Amendment No 208.|You were elected on the 16th of September - as you say perfectly - and this report concerns the elected members of the 13th of June.|
|Vous avez remplacé Mme Reding.|You mentioned the Minutes.|You replaced Ms. Reding.|
|Donc, il y aura un autre rapport qui, je l'espère, confirmera votre mandat.|So, in order to vote against this motion, we are talking about the speech.|So there will be another report that I hope will confirm your mandate.|
|Monsieur le Président, comme nous approchons de la période de Noël, j' aimerais que vous m' offriez un peu de temps !|Mr President, as Korea is our leading since I was king in its words.|Mr President, as we are approaching the Christmas period, I would like you to offer me some time.|
|J' aimerais vous remercier et dissiper un malentendu.|I wish to thank you all commend for another report.|I would like to thank you and dispel a misunderstanding.|
|Le Président a le droit d' autoriser un député à adresser une question à la Commission.|The presidency constitute a significant step forward in support of the Commission.|The President has the right to authorize a Member to address a question to the Commission.|
|Ceci étant dit, Mesdames et Messieurs les Députés, il n'y a plus de point à l'ordre du jour.|Mr President, ladies and gentlemen, some of the moratorium - the adoption here today.|Having said that, ladies and gentlemen, there is no longer any item on the agenda.|
|Le procès-verbal de la présente séance sera soumis à l'adoption du Parlement au début de la prochaine séance.|The Minutes of the previous debate is now very subject with Parliament.|The Minutes of this sitting will be submitted to Parliament for adoption at the beginning of the next sitting.|
|Je donne la parole à M. Manders pour une motion de procédure.|Mr Blak President, thank you for agreeing your proposal.|I call Mr Manders on a point of order.|
|Monsieur le Président, je voudrais profiter de l'occasion pour vous souhaiter, ainsi qu'au Bureau et à tous les collègues, une bonne transition vers la nouvelle année.|Mr President, I wish to draw your attention to something place on Members of the House and all those who have another expected year region next.|Mr President, I would like to take this opportunity to wish you and the Bureau and all your colleagues a good transition to the New Year.|
|Je me permettrai même, bien qu'ils soient absents, de remercier la Commission et le Conseil.|I would really like to see the views of the Commission and the Commission.|I would even like to thank the Commission and the Council, even though they are absent.|
|Je ne rouvrirai pas le débat sur le millénaire, mais je vous souhaite à toutes et à tous et, par-là même, à tous les citoyens européens que nous représentons, une heureuse année 2000.|I cannot sit back to the defeation of the European Union, and I am well aware that consensus in Europe, and not for other countries, we will be able to at least 2000.|I will not reopen the debate on the millennium, but I wish you and all of us, and all the European citizens we represent, a happy 2000.|
|Interruption de la session|Adjournment of the session|Interruption of session|
|Je déclare interrompue la session du Parlement européen.|I declare the session of the European Parliament adjourned.|I declare the session of the European Parliament interrupted.|
|(La séance est levée à 10h50)|(The sitting was closed at 10.50 a.m.)|(The sitting was closed at 10.50 am)|

## Authors
liangstein (lxxhlb@gmail.com, lxxhlb@mail.ustc.edu.cn)

Thanks to the clear written code from [buriburisuri](https://github.com/buriburisuri/ByteNet), which helps me a lot in understanding the structure of the network. 

