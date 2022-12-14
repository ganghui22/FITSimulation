# -*-coding:utf-8-*-
import random

from scipy.stats import skewnorm
import networkx as nx

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PIL import Image
import json
import threading
import time
import jionlp as jio
import calendar
from Graph.expreiment import ground_truth2sample_table,object_time2real_time

location = ['room510', 'room511', 'room512', 'room513', 'room514', 'room515', 'room516']
other_location = ['1号会议室', '2号会议室', '休息室', '茶水间', '1001教室', '1002教室', '1003教室', '讨论区', '其他']
with open('data/Location_list.json', encoding='utf-8') as f:
    location_dict = json.load(f)

total_time_location = 30 * 60
total_time_other = 60  * 60
sigma_location = total_time_location / 2
sigma_other_location = total_time_other / 2

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)


def timetostamp(tss1):
    timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


def stamptotime(stamp):
    timeArray = time.localtime(stamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


def cal_time(a, total_time, u, sigma, t_th):
    a = a
    x = np.linspace(0, total_time, total_time)
    u = u
    sigma = sigma
    l = skewnorm.pdf(x, a, u, sigma)
    return l[t_th]

def y(x, stay_time, early_time=10*60, late_time=30*60):
    sigma_early_time = early_time
    sigma_late_time = late_time
    Normal_early = lambda x: np.multiply(np.power(np.sqrt(2 * np.pi) * sigma_early_time, -1), np.exp(-np.power(x - 0, 2) / (2 * sigma_early_time ** 2)))
    Normal_late = lambda x: np.multiply(np.power(np.sqrt(2 * np.pi) * sigma_late_time, -1), np.exp(-np.power(x - 0, 2) / (2 * sigma_late_time ** 2)))
    if x<0:
        return Normal_early(x)
    elif 0<=x<stay_time:
        return 1-Normal_early(x) - Normal_late(x-stay_time)
    elif x>=stay_time:
        return Normal_late(x-stay_time)

def cal_time_zheng(total_time, u, sigma, t_th):
    x = np.arange(0, total_time, 1)
    pdf = np.exp(-((x - u) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    pdf = (pdf - np.min(pdf)) / (np.max(pdf) - np.min(pdf))
    return pdf[t_th]


from scipy import stats


def cal_time_zhishu(sigma, time_go, time):
    '''
    time is s
    sigma is 参数、
    一般在sigma以后，可能性降为0.5
    '''
    r = 1 / sigma
    X = []
    Y = []
    for x in np.linspace(0, time_go, time_go):
        if x == 0:
            continue
        #   p = r*math.e**(-r*x)  #直接用公式算
        p = stats.expon.cdf(x, scale=1 / r)  # 用scipy.stats.expon工具算,注意这里scale参数是标准差
        X.append(x)
        Y.append(p)
    Y = 1 - (Y - np.min(Y)) / (np.max(Y) - np.min(Y))  #
    return Y[time]







class update():
    def __init__(self, stride):
        self.id = 0
        self.location_total = []
        # print(location_dict)
        for i, xy in location_dict.items():
            self.location_total.append(i.lower())
        self.location_total.append('其他')
        # print(self.location_total)
        self.location_id = {}
        for i in self.location_total:
            self.location_id[i] = self.id
            self.id += 1
        # print(self.location_id)
        # print(list(self.location_id.keys()))
        # print(self.id)
        self.stride = stride  # 采样步数
        self.distribute = [0] * (self.id + 1)
        # print(self.distribute)
        self.lock = threading.Lock()
        # self.condition = threading.Condition()
        self.total_time_o = 0
        self.time = time.asctime(time.localtime(time.time()))
        self.messege = None
        self.waiting_update = []
        with open('data/Person.json', 'r', encoding='utf-8') as f:
            self.ppp = json.load(f)
        # with open('Graph/Graph.json', 'r', encoding='utf-8') as load_f:
        #     self.graph_rel = json.load(load_f)
        self.teacher = ['刘华平', '刘老师']
        self.graph_rel = {}
        self.need_update = {}
        self.virtual_person_location_table = {}
        for p in self.ppp:
            self.graph_rel[p] = {}
            self.graph_rel[p]["rel_base"] = [self.ppp[p]["name"], self.ppp[p]["position"], 1]
            self.graph_rel[p]["rel_now"] = self.distribute.copy()
            # print(self.graph_rel[p]["rel_base"][1].lower())
            # print(self.location_id[self.graph_rel[p]["rel_base"][1].lower()])
            self.graph_rel[p]['rel_now'][self.location_id[self.graph_rel[p]["rel_base"][1].lower()]] = 1
            self.need_update[p] = {}
            self.virtual_person_location_table[p] = [[0] * self.stride for i in range(self.id)]


        self.today_time = time.localtime(time.time())
        self.now_time = timetostamp(
            f"{self.today_time.tm_year}-{self.today_time.tm_mon}-{self.today_time.tm_mday} 08:00:00")

        self.today = timetostamp(
            f"{self.today_time.tm_year}-{self.today_time.tm_mon}-{self.today_time.tm_mday} 00:00:00")
        '''设置时间函数的参数'''
        self.sigma = 100  # 影响最大的可能性---可能性可以乘上sigma
        self.u = 100  # 影响最左边的值
        self.a_time = 10  # 影响左边下降的梯度
        self.person = [p for p in self.ppp]

        self.if_need_change = 0
        self.tmp_graph = {}
        self.event = {}
        self.image = None
        self.mohu = ['一会儿', '一会', '过会']
        self.vitual_envent = {}

    def receive_messege(self, triple, text, label):
        self.triple = triple
        for k, lll in enumerate(self.triple):
            if lll[2] != '':
                result=[]
                mm=object_time2real_time(lll[2], time.time())
                for i in mm:
                    result.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(i)))
                self.triple[k][2] = result
        self.text = text
        self.tmp_dynamic_time_graph()
        self.label = label

    def del_messege(self):
        self.messege = None

    def update_rel(self):
        '''-------------------------------根据json格式更改-接收消息时候的改变--------------------------------'''
        # print('---------go in updat rel--------------')
        # tmp_graph = self.tmp_graph.copy()
        delete_list = []
        for time_dy, messege in self.tmp_graph.items():
            # print(self.tmp_graph)
            if self.now_time >= timetostamp(time_dy):
                for per_event in messege:
                    for p in per_event:
                        ch_person = p[0]
                        ch_location = p[1]
                        # location发生了变化
                        self.graph_rel[ch_person]['rel_now'][self.location_id[ch_location.lower()]] = 1
                        for i in enumerate(self.graph_rel[ch_person]['rel_now'][:self.location_id[ch_location]]):
                            self.graph_rel[ch_person]['rel_now'][i[0]] = 0
                        for j in enumerate(self.graph_rel[ch_person]['rel_now'][self.location_id[ch_location] + 1:]):
                            self.graph_rel[ch_person]['rel_now'][self.location_id[ch_location] + 1 + j[0]] = 0
                        self.need_update[ch_person][time_dy] = [ch_person, ch_location, time_dy]
                delete_list.append(time_dy)
        for o in delete_list:
            del self.tmp_graph[o]
            # time.sleep(5)
        dele_event = []
        for person_name, p in self.event.items():
            for num, person_event in enumerate(p):  # 取出事件
                if self.now_time < timetostamp(person_event[0][2][0]):
                    continue
                else:
                    # print(person_event)
                    for per in person_event:
                        ch_person = per[0]
                        # print(ch_person)
                        ch_location = per[1]
                        # print(ch_location)
                        for i in enumerate(self.graph_rel[ch_person]['rel_now'][:self.location_id[ch_location]]):
                            self.graph_rel[ch_person]['rel_now'][i[0]] = 0
                        for j in enumerate(self.graph_rel[ch_person]['rel_now'][self.location_id[ch_location] + 1:]):
                            ind = self.location_id[ch_location] + 1 + j[0]
                            self.graph_rel[ch_person]['rel_now'][ind] = 0
                        self.graph_rel[ch_person]['rel_now'][self.location_id[ch_location]] = 1
                        self.need_update[ch_person][person_event[0][2][0]] = [ch_person, ch_location,
                                                                              person_event[0][2]]
                    # del tmp1[person_name]
                    dele_event.append(person_name)
        for op in dele_event:
            del self.event[op]

    def nomalize_triple(self, value):
        if value[1] == '你':  # 处理'找你'的情况
            value[1] = self.old_initiator
        if value[0]=='你们':
            return []
        value[0] = self.normalize_name(value[0])
        value[1] = self.normalize_location(value[1])
        if value[1] !='' and value[1] not in self.location_total:
            value[1]='其他'
        if value[1] != '' and value[1] in self.person:  # 当地点是人的时候，对应这个人的办公室
            if self.graph_rel[value[1]]['rel_now'] != None:
                value[1] = self.graph_rel[value[1]]['rel_now'][1]
            else:
                value[1] = self.graph_rel[value[1]]['rel_base'][1]

        if value[1] != '' and '的办公室' in value[1]:
            value[1] = value[1].split('的办公室')[0]
            value[1] = self.graph_rel[value[1]]['rel_base'][1]

        # if value[2] != '':
        #     value[2] = self.get_time(value[2])
        if value[1] != '' and value[2] != '':
            self.per_event.append([value[0], value[1], value[2]])
            # print(self.per_event)
            if value[0] in ['我们', '大家', '咱们', '全体员工', '所有人']:
                self.total_person_flag = 1
        return value

    def normalize_location(self,l):
        if l=='':
            return l
        for j in location:
            if l in j :
                l=j
                break
        return l

    def normalize_name(self, m):
        for p in self.ppp:
            if p in m:
                m = p
                break
            if m in p:
                m=p
                break

        for t in self.teacher:
            if t == m:
                m = '刘老师'
                break
        return m

    def tmp_dynamic_time_graph(self):  # single text
        # self.need_update = {}  # 指的是需要进一步继续时空概率推断的关系
        self.if_need_change = 0
        for i in ['那我们就换', '那我们换',
                  '那咱们就换', '那咱们换',
                  '那咱们改', '那我们就改',
                  '那我们改', '那咱们就改']:
            if i in self.text:
                self.if_need_change = 1
                break
        self.initiator = self.text.split(':')
        self.initiator = self.initiator[0]  # 事件的发起者
        # 修改名字不一样的地方，
        self.initiator = self.normalize_name(self.initiator)

        if self.initiator not in self.event:
            self.event[self.initiator] = []
        self.per_event = []

        if self.if_need_change == 0:  # 发起者发起的新事件，进行事件记录
            try:
                if self.event[self.initiator] != []:
                    time_signal = self.event[self.initiator][0][0][2]
                    # print(self.event[self.initiator][0])
                    if time_signal not in self.tmp_graph:
                        self.tmp_graph[time_signal] = []
                    self.tmp_graph[time_signal].append(self.event[self.initiator][0])
                    del self.event[self.initiator][0]
                self.total_person_flag = 0

                for value in self.triple:
                    value = self.nomalize_triple(value)
                    if self.total_person_flag == 1:  # 把我们更换成人名
                        event_time = value[2]
                        event_location = value[1]
                        self.per_event = []
                        for i in self.person:
                            self.per_event.append([i, event_location, event_time])
                if self.triple != [] and self.per_event != []:
                    self.event[self.initiator].append(self.per_event)
            except:
                # print('-------can not record the triple, drop out----1!!!!--------')
                pass

        # 由于是新的时间，所以把之前的存在的事件就默认已经确定不变，放在self.tmp_graph中，

        elif self.if_need_change == 1:  # 那就xxx
            try:
                for value in self.triple:
                    value = self.nomalize_triple(value)

                    # 时间地点都有
                    if value[2] != '' and value[1] != '':
                        for per in self.event[self.initiator][0]:
                            per[1] = value[1]
                            per[2] = value[2]
                            self.per_event.append(per)
                    # 地点需要更新
                    if value[2] == '' and value[1] != '':
                        for per in self.event[self.initiator][0]:
                            per[1] = value[1]
                            self.per_event.append(per)

                    # 时间需要更新
                    if value[2] != '' and value[1] == '':
                        for per in self.event[self.initiator][0]:
                            per[2] = value[2]
                            self.per_event.append(per)
                self.event[self.initiator].append(self.per_event)
                time_signal = self.event[self.initiator][0][0][2][0]
                if time_signal not in self.tmp_graph:
                    self.tmp_graph[time_signal] = []
                self.tmp_graph[time_signal].append(self.event[self.initiator][0])
                del self.event[self.initiator][0]
            except:
                # print('---------------can not change the time or location----2!!!!!---------------------')
                pass
        elif self.if_need_change == 2:
            try:
                if self.event[self.initiator] != []:
                    time_signal = self.event[self.initiator][0][0][2][0]
                    # print(self.event[self.initiator][0])
                    if time_signal not in self.tmp_graph:
                        self.tmp_graph[time_signal] = []
                    self.tmp_graph[time_signal].append(self.event[self.initiator][0])
                    del self.event[self.initiator][0]
                total_person_flag = 0
                for value in self.triple:
                    value[0] = self.normalize_name(value[0])
                    value[1] = self.normalize_location(value[1])
                    value[1] = self.old_initiator
                    if value[1] != '' and value[1] in self.person:  # 当地点是人的时候，对应这个人的办公室
                        if self.graph_rel[value[1]]['rel_now'] != None:
                            value[1] = self.graph_rel[value[1]]['rel_base'][1]
                        else:
                            value[1] = self.graph_rel[value[1]]['rel_base'][1]
                    if value[1] != '' and '的办公室' in value[1]:
                        value[1] = value[1].split('的办公室')[0]
                        value[1] = self.graph_rel[value[1]]['rel_base'][1]

                    if value[1] != '' and value[2] != '':
                        self.per_event.append([value[0], value[1], value[2]])
                        if value[0] in ['我们', '大家', '咱们', '全体员工', '所有人']:
                            total_person_flag = 1

                if total_person_flag == 1:  # 把我们更换成人名
                    event_time = value[2]
                    event_location = value[1]
                    self.per_event = []
                    for i in self.person:
                        self.per_event.append([i, event_location, event_time])
                if self.triple != [] and self.per_event != []:
                    self.event[self.initiator].append(self.per_event)
            except:
                # print('-------can not record the triple, drop out----3!!!!--------')
                pass

        # time.sleep(0.01)
        self.old_initiator = self.initiator

    # print('====event=========',self.event)

    def update_auto(self):  # 自动的改变
        # try:
        self.update_rel()
        '''按照时间函数更新'''
        need_delete_list = []
        for k, info in self.need_update.items():
            for t, e in info.items():
                # print('for 之后', self.need_update)
                if len(e[2]) == 1:
                    '''设置时间函数的计算参数'''
                    if e[1] in location:
                        self.sigma = sigma_location
                        self.total_time_o = total_time_location
                    elif e[1] in other_location:
                        self.sigma = sigma_other_location
                        self.total_time_o = total_time_other
                    # 更新可能性
                    time_err = self.now_time - timetostamp(e[2][0])

                    if 0 < time_err < self.total_time_o - 1:
                        # tmp_possibillity = cal_time_zheng(self.total_time_o, 0, self.sigma, time_err)
                        tmp_possibillity=y(time_err,stay_time=self.total_time_o)
                        self.graph_rel[k]['rel_now'][self.location_id[e[1]]] = tmp_possibillity  # 更新可能性
                        self.graph_rel[k]['rel_now'][
                            self.location_id[self.graph_rel[k]['rel_base'][1].lower()]] = 1 - tmp_possibillity

                    # 小于阈值的时候，相当于不再存在
                    if time_err > self.total_time_o:  # 这里加上time——err的

                        self.graph_rel[k]['rel_now'][self.location_id[self.graph_rel[k]['rel_base'][1].lower()]] = 1
                        self.graph_rel[k]['rel_now'][self.location_id[e[1]]] = 0
                        # del m[k][t]
                        need_delete_list.append((k, t))

                        # info_detail = '{} come back to office!!!'.format(info[0])

                else:
                    if timetostamp(e[2][0]) < self.now_time < timetostamp(e[2][1]):
                        self.graph_rel[k]['rel_now'][self.location_id[e[1]]] = 1
                        for i in enumerate(self.graph_rel[k]['rel_now'][:self.location_id[e[1]]]):
                            self.graph_rel[k]['rel_now'][i[0]] = 0
                        for j in enumerate(
                                self.graph_rel[k]['rel_now'][self.location_id[e[1]] + 1:]):
                            self.graph_rel[k]['rel_now'][self.location_id[e[1]] + 1 + j[0]] = 0
                    elif self.now_time > timetostamp(e[2][1]):
                        self.graph_rel[k]['rel_now'][self.location_id[self.graph_rel[k]['rel_base'][1].lower()]] = 1
                        self.graph_rel[k]['rel_now'][self.location_id[e[1]]] = 0
                        # del m[k][t]
                        need_delete_list.append((k, t))

        for a, b in need_delete_list:
            del self.need_update[a][b]
        # except Exception as m:
        #     print("update_auto")
        #     print(str(m))
        #     pass
    def simulate_time(self, begin=0, end=0, stride=12):
        s_begin = self.now_time + begin * 60 * 60
        s_end = self.now_time + (12 + end) * 60 * 60
        diff_time = (s_end - s_begin) // self.stride
        # print(diff_time)
        nums = 0  # 第num次采样
        time_copy = s_begin
        total_days = 1
        result = {}
        truth = ground_truth2sample_table(self.label, self.now_time, self.ppp, localtion_dict=location_dict,
                                          start_time=begin, sample_step=stride)
        while total_days > 0:
            result = []
            self.now_time += diff_time
            # print(self.now_time)
            # self.update_rel()
            self.update_auto()
            for pp, table in self.virtual_person_location_table.items():
                for cloumn in range(self.id):  # 地点
                    # print(cloumn,nums)
                    self.virtual_person_location_table[pp][cloumn][nums] = self.graph_rel[pp]['rel_now'][cloumn]
            result.append(self.virtual_person_location_table)
            nums += 1
            if nums == self.stride:
                time_copy += 24 * 60 * 60
                self.now_time = time_copy
                nums = 0
                total_days -= 1
        return self.virtual_person_location_table, truth



def init_graph(Person, Location):
    graph = {}
    for k, person in enumerate(Person):
        m = random.randint(0, len(Location) - 1)
        person_relashionship = {}
        person_relashionship['rel_base'] = [person, Location[m], 1]
        person_relashionship['rel_now'] = None
        graph[person] = person_relashionship

    with open('Graph/graph.json', "w", encoding='utf-8') as f:
        json.dump(graph, f, cls=MyEncoder, ensure_ascii=False)
    return graph


def sample(location):
    per_location = location
    if per_location in location:
        sigma = sigma_location
        total = total_time_location
    else:
        sigma = sigma_other_location
        total = total_time_other
    gfg = np.random.exponential(sigma, total)
    m = total - gfg[random.randint(0, total - 1)]
    while m < 0:
        m = total - gfg[random.randint(0, total - 1)]
    # print(m)
    return m


if __name__ == '__main__':
    # import threading
    #
    # '''-------------初始化他们的办公室--------------'''
    # # Person = ['港晖', '晨峻', '伟华', '刘老师', '袁老师', '刘毅', '姚峰', '侯煊', '小飞',
    # #                '郝伟', '海洋', '春秋', '靖宇', '兴航', '文栋', '兰军', '李老师', '馨竹']
    # # Location = ['Room510', 'Room511', 'Room512', 'Room513', 'Room514', 'Room515', 'Room516']
    # # other_location = ['1号会议室', '2号会议室', '休息室', '茶水间', 'Toilet']
    # # init_graph(Person,Location)
    #
    # # g=Graph()
    # # g.draw(base_graph_rel)
    #
    # m = update()
    # thread1 = threading.Thread(target=m.update_auto)
    # thread2 = threading.Thread(target=m.receive_messege)
    # #
    # thread2.start()
    # thread1.start()
    m = update(12)
    m.simulate_time(False, 0, 0)
