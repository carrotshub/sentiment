# -*- coding: utf-8 -*-
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import jieba



def get_data(file_aname,label):
    x_train=[]
    with open(file_aname,'rb') as f:
        for line in f:
           re=jieba.cut(line.decode('gbk','ignore').strip())
           word_list=[w for w in re]
           document=TaggedDocument(word_list,tags=label)
           x_train.append(document)
    return x_train
    
def train_model(train_set,size=200,epoch_num=1):
    model_dm=Doc2Vec(train_set,min_count=1,window=3,size=size,sample=0.003,negative=5,workers=4)
    model_dm.train(train_set,total_examples=model_dm.corpus_count,epochs=70)
    model_dm.save('./model/model_dm.model')
    return model_dm

def test():
    model_dm=Doc2Vec.load("./model/model_dm.model")
    test_text=['网络', '奇差', '，', '餐厅', '服务', '素质', '奇差', '，', '晚餐', '接完', '帐', '出门', '被', '堵', '着', '又', '要求', '买单', '，', '搞错', '了', '还', '一幅', '理所应当', '的', '样子']
    inferred_vector_dm=model_dm.infer_vector(test_text)
    print(inferred_vector_dm)
    sims=model_dm.docvecs.most_similar([inferred_vector_dm],topn=10)
    return sims


if __name__=='__main__':        
    train_neg=get_data('./data/neg.txt',['neg'])
    train_pos=get_data('./data/pos.txt',['pos'])
    train_set=train_neg+train_pos
    model_dm=train_model(train_set)
    sims=test()
    for count,sim in sims:
            sentence=train_set[count]
            words=''
            for word in sentence[0]:
                words=words +word +' '
            print(words,sim,len(sentence[0]))