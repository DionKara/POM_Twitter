if __name__ == "__main__" :
    
    import pandas as pd
    import numpy as np
    import sys

    input_filename = sys.argv[1]

    #Used this for the rest csv - where there are no column names and data begin straight from the first line
    
    data_in = pd.read_csv(input_filename)

    data_in = data_in.rename(columns={'tweet_text': 'text'})

    
    tweets = pd.DataFrame(columns = ['text'])

    #preprocess used from #1 kernel in kaggle for sentiment140 dataset

    import nltk
    from nltk.stem import WordNetLemmatizer

    from nltk.corpus import stopwords
    import re

    
    HASHTAG_CLEANING_RE = "#\S+"
    MENTION_CLEANING_RE = "@\S+"
    TEXT_CLEANING_RE = "https?:\S+|http?:\S|[^A-Za-z0-9]+"


    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()



    def preprocess(text, lemma=True):
        # Remove link,user and special characters
        text = re.sub(HASHTAG_CLEANING_RE, ' ', str(text).lower())
        text = re.sub(MENTION_CLEANING_RE, ' ', str(text).lower())
        text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in stop_words or token in ['not', 'can']:
                if lemma:
                    tokens.append(lemmatizer.lemmatize(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)

    tweets.text = data_in.text.apply(lambda x: preprocess(x))



    tweets.to_csv('preprocessed_tweets.csv',index=False)


    import fastai
    from fastai.text import *
    from fastai import *
    import regex as re
    import html
    import spacy
    from fastai.text.core import tokenize_texts 
    import collections 
    from collections import Counter
    import pickle


    re1 = re.compile(r'  +')
    BOS = "xxbos"
    FLD = "xxfld"


    def fixup(x):
        x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
            'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
            '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
            ' @-@ ', '-').replace('\\', ' \\ ')
        return re1.sub(' ', html.unescape(x))

    def get_texts(df, n_lbls=0):

        texts = f'\n {FLD} 1 ' + df.iloc[:,n_lbls].astype(str)

        texts = texts.apply(fixup).values.astype(str)


        tokop = tokenize_texts(texts)
        return tokop


    def get_all(df, n_lbls):
        tok = []
        #import pdb
        #pdb.set_trace()
        for i, txt in enumerate(df):
            tok_ = get_texts(txt, n_lbls)
            tok += tok_
        return tok


    chunksize = 24000
    chunk_tweets = pd.read_csv('preprocessed_tweets.csv',chunksize=chunksize)


    # the splitted words of each sentence (not numbers)
    tokens = get_all(chunk_tweets, 0)



    import torch
    from torch.utils.data import DataLoader
    from torch import nn
    import os 
    from torch.autograd import Variable
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



    # Load the original vocab used for training
    [df_train,df_valid,itos, train_tokens, valid_tokens, trn_lm, val_lm] = pickle.load(open('Political social media/dfs_tokens_fastai_NEW.pkl','rb'))


    # Recreate the dictionary used for training to ensure that we use same word indices as in training
    stoi = collections.defaultdict(lambda: 0, { v: k for k, v in enumerate(itos) })



    # recreate the sequences by replacing the words with their indices from the dictionary (stoi_1)
    lm = np.array([ [stoi[o] for o in p] for p in tokens ])

    # add a 'tokens' field in our dataframe with the tokenized sequences
    tweets['tokens'] = lm


    tweets['n_tok'] = tweets['tokens'].apply(len)



    # Padding the sequences to have same length input tweets
    padlen=100 #use the same padlen as in training 
    padding_idx=1

    def pad (x, padlen, padding_idx):
        out=np.ones(padlen)*padding_idx
        out=out.astype(np.int64)
        if len(x)>=padlen:
            out[:]=x[:padlen]
        else:
            out[:len(x)]=x
        return out

    tweets.tokens = tweets.tokens.apply(lambda x: pad(x, padlen, padding_idx))


    tweets.loc[tweets['n_tok'] > padlen, ['n_tok']] = padlen



    # We must define model's class first , to be able to load it.

    n_inp=len(itos)
    n_emb=200 #650
    n_hidden=200#400
    n_layers= 2 # 2
    dropout=0.5 # 0.5
    wd=1e-7
    bidirectional=True
    dropout_e=0.2 # 0.5 - changing to 0.4, 0.3 or any dropout value did not make much difference
    dropout_o=0.5 #0.5
    n_out=1


    class sentiment_classifier (nn.Module):
        def __init__(self,n_inp,n_emb,n_hidden,n_layers,bidirectional,bs,device,dropout_e=0.05,dropout=0.5,\
                     dropout_o=0.5,pretrain_mtx=None,n_out=1,padding_idx=1,n_filters=100,filter_sizes=[3,4,5]):
            super().__init__()
            self.n_inp,self.n_emb,self.n_hidden,self.n_layers,self.bidirectional,self.bs,self.device,self.pretrain_mtx,self.padding_idx=\
                                n_inp,n_emb,n_hidden,n_layers,bidirectional,bs,device,pretrain_mtx,padding_idx
            self.n_out,self.n_filters,self.filter_sizes=n_out,n_filters,filter_sizes
            self.dropout_e,self.dropout,self.dropout_o=dropout_e,dropout,dropout_o

            self.create_architecture()
            if pretrain_mtx is not None:
                print (f'initializing glove with {pretrain_mtx.shape}')
                self.initialize_glove()
            self.init_hidden()
            self.criterion=nn.BCEWithLogitsLoss()

        def set_dropouts(self, dropout, dropout_o, dropout_e):
            self.dropout, self.dropout_o, self.dropout_e = dropout, dropout_o, dropout_e


        def freeze_embedding(self):
            self.encoder.weight.requires_grad=False

        def unfreeze_embedding(self):
            self.encoder.weight.requires_grad=True

        def initialize_glove(self):
            self.encoder.weight.data.copy_(torch.Tensor(self.pretrain_mtx))

        def init_hidden(self):
            # Initialize hidden
            self.hidden=(Variable(torch.zeros(self.n_layers,self.bs,self.n_hidden,requires_grad=False).to(self.device)),
                         Variable(torch.zeros(self.n_layers,self.bs,self.n_hidden,requires_grad=False).to(self.device)))


        def create_architecture(self):
            ###################################
            # Embedding layer - common to both
            ###################################
            self.dropout_enc=nn.Dropout(self.dropout_e)
            self.encoder=nn.Embedding(self.n_inp,self.n_emb,padding_idx=self.padding_idx)

            #######################################
            # For RNN #############################
            #######################################
            # Embedding Layer: Embedding layer just maps each word to an index. n_inp to n_emb mapping is all it does
                # input to this is of shape n_batch * n_seq
             # LSTM Layer
            self.lstm=nn.LSTM(self.n_emb,self.n_hidden,self.n_layers,batch_first=True,dropout=self.dropout,\
                              bidirectional=self.bidirectional)
              # embs are going to be of shape n_batch * n_seq * n_emb
            self.dropout_op=nn.Dropout(self.dropout_o)

            self.avg_pool1d=torch.nn.AdaptiveAvgPool1d(1)
            self.max_pool1d=torch.nn.AdaptiveMaxPool1d(1)


            #######################################
            # For CNN #############################
            #######################################    
            #embedding dimension is the "depth" of the filter and the number of tokens in the sentence is the width.
            self.conv_0=torch.nn.Conv1d (self.n_emb,self.n_filters,kernel_size=self.filter_sizes[0])
            self.conv_1=torch.nn.Conv1d (self.n_emb,self.n_filters,kernel_size=self.filter_sizes[1])
            self.conv_2=torch.nn.Conv1d(self.n_emb,self.n_filters,kernel_size=self.filter_sizes[2])

            self.fc=nn.Linear(len(self.filter_sizes)*self.n_filters+self.n_hidden*4,self.n_out)



        def forward (self,Xb,Xb_lengths):

            ####RNN PORTION
            embs=self.dropout_enc(self.encoder(Xb))
            if Xb.size(0) < self.bs:
                self.hidden=(self.hidden[0][:,:Xb.size(0),:].contiguous(),
                self.hidden[1][:,:Xb.size(0),:].contiguous())
            packed_embs = pack_padded_sequence(embs,Xb_lengths.cpu(),batch_first=True, enforce_sorted=False)
            lstm_out,(hidden,cell)=self.lstm(packed_embs)
            lstm_out,lengths=pad_packed_sequence(lstm_out,batch_first=True)
            hidden = self.dropout_op(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            avg_pool=self.avg_pool1d(lstm_out.permute(0,2,1)).view(Xb.size(0),-1)
            max_pool=self.max_pool1d(lstm_out.permute(0,2,1)).view(Xb.size(0),-1)

            #CNN Portion
            new_embs=embs.permute(0,2,1)        
            conved_0=torch.relu(self.conv_0(new_embs))
            conved_1=torch.relu(self.conv_1(new_embs))
            conved_2=torch.relu(self.conv_2(new_embs)) 
            max_pool1d=torch.nn.MaxPool1d(conved_0.shape[2])
            pooled_0=max_pool1d(conved_0).squeeze(2)
            max_pool1d=torch.nn.MaxPool1d(conved_1.shape[2])
            pooled_1=max_pool1d(conved_1).squeeze(2)
            max_pool1d=torch.nn.MaxPool1d(conved_2.shape[2])
            pooled_2=max_pool1d(conved_2).squeeze(2)
            cat_cnn = self.dropout_op(torch.cat([pooled_0,pooled_1,pooled_2],dim=1))

            ## Concatenate
            big_out=torch.cat([cat_cnn,hidden,max_pool],dim=1)
            preds=self.fc(big_out)

            preds = torch.sigmoid(preds.view(-1))


            return preds


    # Load the model

    COMBO_PATH = "Political social media/combo model_FINE"

    model_sentiment = torch.load (f'{COMBO_PATH}/model_sentiment')

    device="cuda:0"
    model_sentiment = model_sentiment.to(device)


    # input - df: a Dataframe, chunkSize: the chunk size
    # output - a list of DataFrame
    # purpose - splits the DataFrame into smaller chunks
    def split_dataframe(df, chunk_size = 100): 
        chunks = list()
        num_chunks = len(df) // chunk_size + 1
        for i in range(num_chunks):
            chunks.append(df[i*chunk_size:(i+1)*chunk_size])
        return chunks


    iterable = split_dataframe(tweets)



    # Get predictions on our new data

    y_pred_1 = np.zeros(tweets.shape[0])
    k=0

    for data in iterable:
        data = data.reset_index(drop=True)
        x = data['tokens']
        x_len = data['n_tok']

        x = torch.tensor(x)
        x_len = torch.tensor(x_len)

        x = x.to(device)
        x_len = x_len.to(device)

        y_pred = model_sentiment(x, x_len)

        y_pred = y_pred.to("cpu")
        y_pred = y_pred.detach().numpy()

        y_pred_1[k:k+100] = y_pred
        k+=100

        del x
        del x_len

        torch.cuda.empty_cache()




    import tensorflow
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    from tensorflow.keras.layers import (
        Dense,
        LSTM,
        Embedding,
        SpatialDropout1D,
    )
    from tensorflow.keras.models import (
        Model,
        load_model,
        Sequential
    )
    from tensorflow.keras.callbacks import ModelCheckpoint
    from keras.models import model_from_json
    from numpy import array


    # Tokenization and padding
    
    with open('Youtube Comments/tokenizer.pickle', 'rb') as handle:
        tk = pickle.load(handle)

    tokenized_tweets = tk.texts_to_sequences(tweets['text'])

    max_len = 200 # Calculate as max in dataset
    padded_tweets = pad_sequences(tokenized_tweets, maxlen=max_len, padding='post')



    json_file = open('Youtube Comments/rnn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)



    #Load best model
    x = padded_tweets

    model.load_weights("Youtube Comments/rnn_model_732.h5")
    
    y_pred_2 = model.predict(x, batch_size=1000)
    
    y_pred_2 = y_pred_2.flatten()


    del x
    del model
    tensorflow.keras.backend.clear_session()


    



    # Load the original vocab used for training
    [df_train,df_valid,itos, train_tokens, valid_tokens, trn_lm, val_lm] = pickle.load(open('Tweets with sarcasm and irony - Binary/dfs_tokens_fastai_NEW.pkl','rb'))


    # Recreate the dictionary used for training to ensure that we use same word indices as in training
    stoi = collections.defaultdict(lambda: 0, { v: k for k, v in enumerate(itos) })


    # recreate the sequences by replacing the words with their indices from the dictionary (stoi_1)
    lm = np.array([ [stoi[o] for o in p] for p in tokens ])

    # add a 'tokens' field in our dataframe with the tokenized sequences
    tweets['tokens'] = lm



    tweets['n_tok'] = tweets['tokens'].apply(len)



    # Padding the sequences to have same length input tweets
    padlen=33 #use the same padlen as in training 
    padding_idx=1

    def pad (x, padlen, padding_idx):
        out=np.ones(padlen)*padding_idx
        out=out.astype(np.int64)
        if len(x)>=padlen:
            out[:]=x[:padlen]
        else:
            out[:len(x)]=x
        return out

    tweets.tokens = tweets.tokens.apply(lambda x: pad(x, padlen, padding_idx))


    tweets.loc[tweets['n_tok'] > padlen, ['n_tok']] = padlen






    # Load the model

    COMBO_PATH = "Tweets with sarcasm and irony - Binary/combo model_FINE"

    model_sentiment = torch.load (f'{COMBO_PATH}/model_sentiment')

    device = "cuda:0"
    model_sentiment = model_sentiment.to(device)


    iterable = split_dataframe(tweets)


    # Get predictions on our new data

    y_pred_3 = np.zeros(tweets.shape[0])
    k=0

    for data in iterable:
        data = data.reset_index(drop=True)
        x = data['tokens']
        x_len = data['n_tok']

        x = torch.tensor(x)
        x_len = torch.tensor(x_len)

        x = x.to(device)
        x_len = x_len.to(device)

        y_pred = model_sentiment(x, x_len)

        y_pred = y_pred.to("cpu")
        y_pred = y_pred.detach().numpy()

        y_pred_3[k:k+100] = y_pred
        k+=100

        del x
        del x_len

        torch.cuda.empty_cache()


    # Load the original vocab used for training
    [df_train,df_valid,itos, train_tokens, valid_tokens, trn_lm, val_lm] = pickle.load(open('OLID/dfs_tokens_fastai.pkl','rb'))


    # Recreate the dictionary used for training to ensure that we use same word indices as in training
    stoi = collections.defaultdict(lambda: 0, { v: k for k, v in enumerate(itos) })



    # recreate the sequences by replacing the words with their indices from the dictionary (stoi_1)
    lm = np.array([ [stoi[o] for o in p] for p in tokens ])

    # add a 'tokens' field in our dataframe with the tokenized sequences
    tweets['tokens'] = lm


    tweets['n_tok'] = tweets['tokens'].apply(len)


    # Padding the sequences to have same length input tweets
    padlen=45 #use the same padlen as in training 
    padding_idx=1

    def pad (x, padlen, padding_idx):
        out=np.ones(padlen)*padding_idx
        out=out.astype(np.int64)
        if len(x)>=padlen:
            out[:]=x[:padlen]
        else:
            out[:len(x)]=x
        return out

    tweets.tokens = tweets.tokens.apply(lambda x: pad(x, padlen, padding_idx))


    tweets.loc[tweets['n_tok'] > padlen, ['n_tok']] = padlen






    # Load the model

    COMBO_PATH = "OLID/combo model"

    model_sentiment = torch.load (f'{COMBO_PATH}/model_sentiment')

    device = "cuda:0"
    model_sentiment = model_sentiment.to(device)


    iterable = split_dataframe(tweets)


    # Get predictions on our new data

    y_pred_4 = np.zeros(tweets.shape[0])
    k=0

    for data in iterable:
        data = data.reset_index(drop=True)
        x = data['tokens']
        x_len = data['n_tok']

        x = torch.tensor(x)
        x_len = torch.tensor(x_len)

        x = x.to(device)
        x_len = x_len.to(device)

        y_pred = model_sentiment(x, x_len)

        y_pred = y_pred.to("cpu")
        y_pred = y_pred.detach().numpy()

        y_pred_4[k:k+100] = y_pred
        k+=100

        del x
        del x_len

        torch.cuda.empty_cache()


    # create a matrix n_tweets * 6 (6 = 5(descriptor) + 1(date/time) )
 

    
    data_in['bias-1'] = y_pred_1
    data_in['positive-1'] = y_pred_2
    data_in['figurative-0'] = y_pred_3
    data_in['offensive-1'] = y_pred_4
    


    size = len(input_filename)

    output_filename = input_filename[:size - 4] + "_results.csv"

    data_in.to_csv(output_filename, index=False)












