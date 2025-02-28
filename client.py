# -*- coding: utf-8 -*-


import numpy as np

class client(object):
    def __init__(self,
                 index,
                 thorizon,
                 narms,
                 nclients,
                 palpha,
                 fp):
        self.T = thorizon # rounds
        self.id = index # client id
        self.K = narms # number of arms
        self.M = nclients # number of clients
        self.alpha = palpha # the hyperparameter
        self.fp = fp # designated function

        self.p = 1
        self.local_set = set(np.arange(self.K)) # testing arms locally
        self.global_set = set(np.arange(self.K)) # testing arms globally
        self.local_mean = np.zeros(self.K) # local mean of each arm
        self.global_mean = np.zeros(self.K) # global mean of each arm
        self.mixed_mean = np.zeros(self.K) # mixed mean of each arm (alpha*local_mean+(1-alpha)*global_mean)
        self.reward = np.zeros(self.K) # reward of each arm

        self.pull = np.zeros(self.K) # chosen arm
        self.p_length = self.fp(self.p) # f(p)
        self.Fp = 0

        self.fphase = 0
        self.gphase = 0

        self.F = -1
        self.l_exploration = False
        self.g_exploration = False

    def play(self):
        if self.fphase < np.ceil((1-self.alpha)*self.p_length)*len(self.global_set): #global exploration
            # print("loc", self.p)
            # play = list(self.global_set)[self.fphase%len(self.global_set)]
            play = list(self.global_set)[int(self.fphase//(np.ceil((1-self.alpha)*self.p_length)))]
            self.fphase += 1

        elif self.gphase < np.ceil(self.M*self.alpha*self.p_length)*len(self.local_set): #local exploration
            # print("glob", self.p)
            # play = list(self.local_set)[self.gphase%len(self.local_set)]
            play = list(self.local_set)[int(self.gphase//(np.ceil(self.M*self.alpha*self.p_length)))]
            self.gphase += 1

        else: #exploitation phase
            if self.l_exploration is True:
                play = self.F
            else:
                play = np.argmax(self.alpha*self.local_mean+(1-self.alpha)*self.global_mean)

        return play

    def reward_update(self,play,obs):
        self.reward[play] += obs
        self.pull[play] += 1

    def local_mean_update(self):
        #print('global',self.fphase,np.ceil((1-self.alpha)*self.p_length)*len(self.global_set))
        #print('local',self.gphase,np.ceil(self.M*self.alpha*self.p_length)*len(self.local_set))
        if self.g_exploration is False and self.fphase >= np.ceil((1-self.alpha)*self.p_length)*len(self.global_set) and self.gphase >= np.ceil(self.M*self.alpha*self.p_length)*len(self.local_set):
            self.local_mean = self.reward/self.pull
            #print("local_mean",self.local_mean, "phase", self.p)
            return True, self.local_mean
        else:
            return False, 0

    def global_mean_update(self,global_stat):
        self.global_mean = global_stat
        self.mixed_mean = self.alpha*self.local_mean+(1-self.alpha)*self.global_mean

    def local_set_update(self):
        Ep = set()
        self.Fp += self.p_length
        conf_bound = np.sqrt(np.log(self.T)/(self.M*self.Fp))
        for i in list(self.local_set):
            if self.mixed_mean[i]+conf_bound < max(self.mixed_mean-conf_bound):
                Ep.add(i)
        self.local_set = self.local_set-Ep

        if len(self.local_set) == 1 and self.l_exploration is False:
            self.l_exploration = True
            self.F = list(self.local_set)[0]
            #print("player", self.id,  " fixate",self.F)
            self.local_set = set()
        #print("player", self.id, " local-set:",self.local_set)
        return self.local_set

    def global_set_update(self,global_set):
        self.global_set = global_set
        self.p +=1
        self.p_length = self.fp(self.p)
        #print("p-length",self.p_length)
        self.fphase = 0
        self.gphase = 0
        self.g_exploration = (len(self.global_set)==0)