import gym
import numpy as np
import torch

# Bits Generation Environment
class Bits_Env(gym.Env):
    '''
    Input:
        n: length of sequence
        reward_type: refer to README for details
        group: number of tokens as a whole
    '''
    def __init__(self,n,reward_type=0,group=1):
        super().__init__()
        self.len=(n//group)*group # length must be divided exactly by group
        self.target=np.random.randint(2**group,size=self.len)
        self.pad=-1 # token has not been generated
        self.state=np.array([self.pad]*self.len)
        self.reward_type=reward_type
        self.pos=0 # the position where to generate token in next step

    '''
    Description: 
        reset the whole environment including 'state' and 'target'
    Input:
        n: the begining position of state sequence  (i.e. state[:n]=target[:n])
        seed & options: parameters of 'reset' function in gym.Env
    Output: 
        a dictionary including 'state' and 'target'
    '''
    def reset(self,n=0,seed=None,options= None):
        super().reset(seed=seed,options=options)
        self.pos=n
        self.target=np.random.randint(2,size=self.len)
        self.state=self.target.copy()
        self.state[self.pos:]=self.pad
        return {'state':torch.tensor(self.state),'target':torch.tensor(self.target)}

    '''
    Description: 
        reset state sequence
    Input:
        n: the begining position of state sequence  (i.e. state[:n]=target[:n])
    Output: 
        a dictionary including 'state' and 'target'
    '''
    def reset_state(self,n=0):
        self.state=self.target.copy()
        self.pos=n
        self.state[self.pos:]=self.pad
        return {'state':torch.tensor(self.state),'target':torch.tensor(self.target)}

    '''
    Input:
        action: the generated token in current step
    Output: 
        a dictionary including 'state' and 'target', reward, and is_done
    '''
    def step(self, action):
        self.state[self.pos]=action
        self.pos+=1
        reward=self.get_reward()
        terminated=(self.pos==self.len)
        return {'state':torch.tensor(self.state),'target':torch.tensor(self.target)},reward,terminated

    def get_reward(self):
        return self._compute_reward(self.target,self.state,self.pos-1)

    '''
    Input:
        target: target sequence
        state: state sequence
        action: the generated token in current step
    Output: 
        reward
    '''
    def compute_reward(self,target,state,action):
        pos=torch.where(state==self.pad)[0][0].item()
        state=state.clone()
        state[pos]=action
        return self._compute_reward(target,state,pos),(state==target).all()

    '''
    Input:
        target: target sequence
        state: state sequence
        pos: the position of the token generated in the last step
    Output: 
        reward
    '''
    def _compute_reward(self,target,state,pos=0):
        if self.reward_type==0:
            return float((state==target).all())
        elif self.reward_type==1:
            if state[-1]==self.pad:
                return 0
            else:
                return torch.sum(torch.tensor(state==target)).item()/self.len
        elif self.reward_type==2:
            return torch.sum(torch.tensor(state == target)).item() / self.len
        elif self.reward_type==3:
            return float(target[pos]==state[pos])/self.len
        elif self.reward_type==4:
            if (self.target[:pos]==self.state[:pos]).all():
                if self.target[pos]==self.state[pos]:
                    return float((state == target).all())
                else:
                    return -1
            else:
                return 0
        elif self.reward_type==5:
            if (self.target[:pos]==self.state[:pos]).all():
                if self.target[pos]==self.state[pos]:
                    return float(target[pos]==state[pos])/self.len
                else:
                    return -1
            else:
                return float(target[pos]==state[pos])/self.len
