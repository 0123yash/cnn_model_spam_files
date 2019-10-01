from gensim.models import Word2Vec
import pandas as pd

#csvFilePath = "/data/ourdata/yash_work/commentsonly_hindi_eng_cleaned_dot_consec_latin.csv"
csvFilePath = "/data/ourdata/yash_work/english_cleaned/commentsonly_eng_consec_3Jan_cleaned.csv"

print('reading csv : ', csvFilePath)

df_csv = pd.read_csv(csvFilePath, names=['C_T'])
list_cmts = df_csv['C_T'].tolist()

print('preparing feed for english word2vec with consec check applied for any character')

word2vec_feed = [(str(sent)).split(" ") for sent in list_cmts]

print('length word2vec_feed : ', len(word2vec_feed))

#model = Word2Vec(word2vec_feed, size=300, window=10, min_count=10, workers=24, negative=15, hs=0, sample=0.00001, sg=0)
model = Word2Vec(word2vec_feed, size=300, window=10, min_count=5, workers=24, negative=15, hs=0, sample=0.00001, sg=0)

print('word2vec model created')
print('saving the word2vec model')

#model.save("hindi_eng_comments_min10_consec.model")
model.save("eng_comments_min5_dot_consec.model")
model.wv.save_word2vec_format("eng_comments_min5_dot_consec_word2vec.bin",  binary=True)

print('size of vocab : ' , len(model.wv.vocab))
