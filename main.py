# -*-coding:utf-8-*-
import json

import NlpToolKit.Chinese.DialoguePrediction as extract_triple
import Graph.DynamicSpaceTimeGraph_SIMU as graph
import numpy as np
import time
from Graph.expreiment import get_acc_one_sample, get_one_sample_precision_and_recall_with_no_resident, \
    get_one_sample_precision_and_recall


class deal():
    def __init__(self):
        self.triple = extract_triple.DialoguePrediction()

    def caluate(self, stride, begin, end, messege, label):
        self.update_graph = graph.update(stride)
        for k,i_per in enumerate(messege):
            for i in i_per:
                m.dynamic_space_time_graph(i, label[k])

        pre, truth = self.update_graph.simulate_time(begin=begin, end=end, stride=stride)
        res = get_one_sample_precision_and_recall_with_no_resident(pre, truth, 0.2)
        return res

    def dynamic_space_time_graph(self, text, label):
        triple = self.triple(text)
        self.update_graph.receive_messege(triple, text, label)
        self.if_need_change = 0

    def person_get_location(self, person_name):
        if person_name in self.update_graph.graph_rel:
            person_message = self.update_graph.graph_rel[person_name]
            a = person_message['rel_now']
            max_loc_id = np.where(a == np.max(a))[0][0]
            return self.update_graph.location_total[max_loc_id]
        else:
            return None

    def get_graph(self):
        return self.update_graph.image

    def get_now_time(self):
        now_time = stamptotime(self.update_graph.now_time)
        return now_time


def stamptotime(stamp):
    timeArray = time.localtime(stamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime

#
# with open('experiment.json', encoding='utf-8') as f:
#     dataset = json.load(f)
#     for sample in dataset:
#         sample/
# exit()

if __name__ == '__main__':
    m = deal()
    with open('experiment.json', encoding='utf-8') as f:
        dataset = json.load(f)

    sum = 0
    nums = len(dataset)
    n = 1
    r_sum, p_sum, f1_sum = 0, 0, 0
    messege=[]
    label=[]
    num=5
    total=num
    for i in dataset:
        messege.append(i['dialogue'])
        label.append(i['label'])
        idx_dia = i['index']
        total-=1
        if total==0:
            r, p, f1 = m.caluate(720, 0, 0, messege, label)
            messege=[]
            label=[]
            total=num
            if (r, p, f1) == (0, 0, 0):
                continue
            r_sum += r
            p_sum += p
            # f1_sum += f1
            print(r, p, f1)
            a_r=r_sum/n
            a_p=p_sum/n
            f1_a=2*a_p*a_r/(a_p+a_r)
            print(a_r, a_p, f1_a, idx_dia)
            n += 1





