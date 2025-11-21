import numpy as np
from collections import defaultdict
import json

class DataLoad:
    def __init__(self, data_dir):
        #self.reg2id = self.load_reg(data_dir)
        self.ent2id, self.rel2id, self.kg_data = self.load_kg(data_dir)
        self.nstation = len(self.ent2id)

        print('types of station=%d, number of relations=%d, type of relations=%d' % (len(self.ent2id), len(self.kg_data), len(self.rel2id)))
        #print('region num={}'.format(len(self.reg2id)))
        print('load finished..')
   
    def load_kg(self, data_dir):    #id 表示的ukg
        ent2id, rel2id = {}, {}
        kg_data_str = []
        with open(data_dir, 'r') as f:
            for line in f.readlines(): 
                h,r,t = line.strip().split()
                kg_data_str.append((h,r,t))
        #ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])))
        ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])), \
              key=lambda x: (0, int(x.split('station')[1])) if 'station' in x and len(x.split('station')) > 1 else (1, x))
        rels = sorted(list(set([x[1] for x in kg_data_str])))

        print('------ents-------')
        print(ents)
        print(rels)
        #for i, x in enumerate(ents):
        #    try:
        #        ent2id[x]
        #    except KeyError:
        #       ent2id[x] = len(ent2id)
        ent2id = dict([(x, i) for i, x in enumerate(ents)])
        rel2id = dict([(x, i) for i, x in enumerate(rels)])

        print('------ent2id-----')
        print(ent2id)
        print(rel2id)
        kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in kg_data_str]
        
        return ent2id, rel2id, kg_data
    
        