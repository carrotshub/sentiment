# -*- coding: utf-8 -*-
import jieba 


file_name=['neg','pos']

for tag in file_name:
    with open(f'./data/train_{tag}.dat','w',encoding='utf-8') as ff:
        with open(f'./data/{tag}.txt','rb') as f:
            for line in f:
                line=f.readline().decode('gb2312','ignore').strip()
                filter_chara="\r\n\t，。；！,.:;：、“”‘’"
                trans_dict=dict.fromkeys((ord(_) for _ in filter_chara),'')
                line=line.translate(trans_dict)
                #word segment
                it=jieba.cut(line,cut_all=False)
                _=[word.strip() for word in it]
                ff.write(' '.join(_)+'\n')
            