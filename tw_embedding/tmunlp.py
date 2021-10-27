import json
import gensim
import pandas as pd
from gensim.models.word2vec import Word2Vec
import math

# 正體中文詞嵌入向量 WORD2VEC
# 300 維，200 維，100 維，50 維模型壓縮檔（以 GENSIM PYTHON 套件訓練）
# 引用詞嵌入模型檔之範例：
# http://nlp.tmu.edu.tw/word2vec/index.html
# vocab_size: 1,723,175 (1.7M)
# size: 2,084,486,400 (2GB)
fname = 'tmunlp_1.6B_WB_300dim_2020v1'  # size: 2,084,486,400 (2GB)
out_vocab_size = 500000
out_vocab_name = '500k'
inp_fname = f'{fname}.bin'
out_meta_tsv =   f'{fname}_{out_vocab_name}_meta.tsv'
out_vector_tsv = f'{fname}_{out_vocab_name}_vector.tsv'
out_config_json = f'{fname}_{out_vocab_name}_projector_config.json'
vec_url = f'http://mail.im.tku.edu.tw/~seke/tmp/{out_vector_tsv}'
meta_url = f'https://raw.githubusercontent.com/sekewei/projector_config/master/tw_embedding/{out_meta_tsv}'

path = inp_fname
model = gensim.models.KeyedVectors.load_word2vec_format(path,
                                                        unicode_errors='ignore', 
                                                        binary=True)
vocab_size = len(model)
vocab_dim = model[model.index_to_key[0]].shape[0]
vocab_index = list(range(vocab_size))
print(vocab_index[:100])

for index in range(100):
    word = model.index_to_key[index]
    print(f'{index}: {word}, len={len(word)}')

vocab_index.sort(key=lambda index: len(model.index_to_key[index]))
print(vocab_index[:100])
#exit(0)

'''
{
  "embeddings": [
    {
      "tensorName": "老人と海",
      "tensorShape": [
        3147,
        300
      ],
      "tensorPath": "https://raw.githubusercontent.com/sekewei/projector_config/master/old_man_and_sea/jp_vector.tsv",
      "metadataPath": "https://raw.githubusercontent.com/sekewei/projector_config/master/old_man_and_sea/jp_meta.tsv"
    }
  ]
}
'''
json_obj = dict()
embed1 = dict()
embed1['tensorName'] = fname
embed1['tensorShape'] = [out_vocab_size, vocab_dim]
embed1['tensorPath']  = vec_url
embed1['metadataPath'] = meta_url
embed_list = [embed1]
json_obj['embeddings'] = embed_list
with open(out_config_json, 'w', encoding='utf-8') as f:
    json.dump(json_obj, f, ensure_ascii=False, indent=4)

#vocab_size = len(model.index_to_key)
print(f'a total of {vocab_size} words.') # vocab_size: 1,723,175 (1.7M)


meta_list = []
vector_list = []
count = 0
for i in range(vocab_size):
    word = model.index_to_key[vocab_index[i]]
    vector = model[word]
    if len(word)<2:
        continue

    #df_meta.loc[len(df_meta)] = [word]
    #df_vector.loc[len(df_meta)] = list(vector)
    meta_list.append([word, len(word)])
    vector_list.append(list(vector))
    count = count + 1
    if count >= out_vocab_size:
        break
    #print(f'{i}\t{word}\t{len(vector)}\t{vector}')
    #print(f'{i}\t{word}')
#print(model.index2word)
#print(len(model))

df_meta = pd.DataFrame(meta_list, columns=['word', 'len'])
df_vector = pd.DataFrame(vector_list)

print(f'{df_meta.shape}')
print(f'{df_vector.shape}')

with open(out_meta_tsv,'w') as write_tsv:
    write_tsv.write(df_meta.to_csv(sep='\t', index=False))

with open(out_vector_tsv,'w') as write_tsv:
    write_tsv.write(df_vector.to_csv(sep='\t', index=False, header=False))
