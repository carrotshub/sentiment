# -*- coding: utf-8 -*-
from gensim.models.doc2vec import Doc2Vec
def test1():
    model_dm=Doc2Vec.load("./model/model_dm.model")
    test_text=['网络', '奇差', '，', '餐厅', '服务', '素质', '奇差', '，', '晚餐', '接完', '帐', '出门', '被', '堵', '着', '又', '要求', '买单', '，', '搞错', '了', '还', '一幅', '理所应当', '的', '样子']
    inferred_vector_dm=model_dm.infer_vector(test_text)
    print(inferred_vector_dm)
    sims=model_dm.docvecs.most_similar([inferred_vector_dm],topn=10)
    return sims


test1()