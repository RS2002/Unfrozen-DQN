import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# buffer to store and replay history
class Buffer:
    '''
    Input:
        size: buffer size
    '''
    def __init__(self,size=100):
        self.size=int(size)
        self.state=[None]*self.size
        self.action=[None]*self.size
        self.reward=[None]*self.size
        self.next_state=[None]*self.size
        self.target=[None]*self.size
        self.done=[None]*self.size
        self.index=0
        self.full=False

    '''
    Description: 
        store elemetn
    Input:
        element: a dictionar including: state,action,reward,next_state,target,done
    '''
    def store(self,element):
        self.state[self.index]=element['state']
        self.action[self.index]=element['action']
        self.reward[self.index]=element['reward']
        self.next_state[self.index]=element['next_state']
        self.target[self.index]=element['target']
        self.done[self.index]=float(element['done'])
        self.index+=1
        if self.index==self.size:
            self.full=True
            self.index=0

    # clear the buffer
    def clean(self):
        self.state=[None]*self.size
        self.action=[None]*self.size
        self.reward=[None]*self.size
        self.next_state=[None]*self.size
        self.target=[None]*self.size
        self.done = [None] * self.size
        self.index=0
        self.full=False

    '''
    Description: 
        Experience Replay
    Input:
        batch_size
    Output:
        state,target,action,reward,next_state,done
    '''
    def sample(self,batch_size):
        sample_index=random.sample(range(self.size),batch_size)
        return torch.stack([self.state[i] for i in sample_index]),torch.stack([self.target[i] for i in sample_index]),torch.tensor([self.action[i] for i in sample_index]),torch.tensor([self.reward[i] for i in sample_index]),torch.stack([self.next_state[i] for i in sample_index]),torch.tensor([self.done[i] for i in sample_index])

    def is_full(self):
        return self.full


class Agent_HER:
    '''
    Input:
        env
        policy_network
        target_network
        replay_buffer
        gamma: discount factor
        lr: learning rate
        batch_size
        eps_start: start probability of ε-greedy
        eps_final: mininum probability of ε-greedy
        eps_decay: decay factor of ε-greedy
        device
        HER: the parameter in HER (HER==None means not using HER)
        my_version: whether to use HER of my version
    '''
    def __init__(self,env,policy_network,target_network=None,replay_buffer=None,gamma=0.99,lr=1e-3,batch_size=32,eps_start=1,eps_final=0.1,eps_decay=0.995,device=torch.device("cpu"),HER=None,my_version=False):
        self.env=env
        self.device=device
        self.policy_network=policy_network.to(self.device)
        self.target_network=target_network
        if self.target_network is not None:
            self.target_network=self.target_network.to(self.device)
            self.update_target()
        self.replay_buffer=replay_buffer
        self.gamma=gamma
        self.eps_start=eps_start
        self.eps_final=eps_final
        self.eps_decay=eps_decay
        self.eps=self.eps_start
        self.loss_func=nn.MSELoss()
        self.optimizer=optim.Adam(self.policy_network.parameters(),lr=lr)
        self.batch_size=batch_size
        self.HER=HER
        self.my_version=my_version

    # select action by ε-greedy
    def select_action(self,eps,state,target):
        rand=torch.rand(1).item()
        if rand<eps: # choose action randomly
            return math.ceil(rand/(eps/(2**self.env.group)))
        else: # choose action with the max Q-value
            return self.policy_network(torch.cat([state,target],dim=-1).to(self.device)).max(dim=-1)[1].item()

    '''
    Description: 
        train policy network
    Input:
        episode: maximum episode
        update_episode: number of episodes to update target network
        save_path: path to save model parameters
        save_log: path to save log file
    '''
    def train(self,episode=50000,update_episode=10,save_path=None, save_log="./log"):
        self.log={"episode":[],"success rate":[]}
        for i in range(1,int(episode)+1):
            obs=self.env.reset()
            done=False
            loss=[]
            # store elements in current episode
            state_episode=[None]*self.env.len
            next_state_episode=[None]*self.env.len
            action_episode=[None]*self.env.len
            reward_episode=[None]*self.env.len
            done_episode=[None]*self.env.len
            index=0
            # explore in environment
            while not done:
                self.policy_network.train()
                state=obs['state'].to(self.device)
                target=obs['target'].to(self.device)
                action=self.select_action(self.eps,state,target)
                obs,reward,done=self.env.step(action)
                # store elements
                self.replay_buffer.store({'state':state.to(self.device),'target':target.to(self.device),'reward':reward,'next_state':obs['state'].to(self.device),'action':action,'done':done})
                state_episode[index]=state.to(self.device)
                next_state_episode[index]=obs['state'].to(self.device)
                reward_episode[index]=reward
                done_episode[index]=done
                action_episode[index]=action
                index+=1
            # original HER
            if self.HER is not None and self.my_version==False:
                for j in range(self.env.len):
                    for k in range(self.HER):
                        new_target = next_state_episode[random.randint(j, self.env.len-1)] # choose 'new target' from 'next state'
                        reward,done = self.env.compute_reward(new_target,state_episode[j],action_episode[j]) # compute new reward in new target
                        self.replay_buffer.store(
                            {'state': state_episode[j], 'target': new_target, 'reward': reward, 'next_state': next_state_episode[j],
                             'action': action_episode[j], 'done': done})
            # only when the buffer is full, the network will train
            if self.replay_buffer.is_full():
                loss.append(self.update_policy())
            if i % update_episode==0:
                self.update_target()
                # evaluate current model
                success_rate=0
                for k in range(100):
                    success_rate+=self.eval()
                success_rate/=100
                # write log & save model
                line="Episode:{}  Success Rate:{}  Loss:{}".format(i,success_rate,np.average(loss))
                if save_log is not None:
                    print(line)
                    with open(save_log+".txt", 'a') as outfile:
                        outfile.write(line+"\n")
                self.log["episode"].append(i)
                self.log["success rate"].append(success_rate)
                loss.clear()
                if save_path is not None:
                    self.save(save_path)
                    print("Model Saved")
                # when the acc is larger than 0.95 for 10 continuous epoches, the training will stop
                if len(self.log["episode"])>10 and (np.array(self.log["success rate"][-10:])>0.95).all():
                    break
            # ε decay
            if self.eps*self.eps_decay>self.eps_final:
                self.eps*=self.eps_decay
        np.save(save_log+".npy",self.log)

    '''
    Description: 
        evaluate current model
    Output:
        success or fail
    '''
    def eval(self):
        self.policy_network.close_grid() # to save GPU memory
        self.policy_network.eval()
        obs = self.env.reset()
        done = False
        while not done:
            action = self.select_action(0, obs['state'], obs['target'])
            obs, reward, done = self.env.step(action)
        return float((obs['state']==obs['target']).all())

    def update_policy(self):
        self.policy_network.open_grid()
        #sample batch
        state, target, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        if self.HER is not None and self.my_version: # my version : just add sample with reward=1
            _,_,_,_,new_target,_=self.replay_buffer.sample(self.HER)
            for i in range(self.HER):
                target[i]=next_state[i]
                reward[i]=1
                done[i]=True
                self.replay_buffer.store({'state':state[i],'target':target[i],'reward':reward[i],'next_state':next_state[i],'action':action[i],'done':done[i]})
        state=state.to(self.device)
        target=target.to(self.device)
        next_state=next_state.to(self.device)
        reward=reward.to(self.device)
        done=done.to(self.device)
        #compute real q_value and target q-value
        q_values=self.policy_network(torch.cat([state,target],dim=-1))
        q_value=torch.tensor([0.0]*len(action))
        for i in range(len(action)):
            q_value[i]=q_values[i,action[i]]
        next_q_value=self.target_network(torch.cat([next_state,target],dim=-1).to(self.device)).max(dim=-1)[0]
        target_q_value=reward+self.gamma*next_q_value * (1 - done)
        q_value=q_value.to(self.device)
        target_q_value=target_q_value.to(self.device)
        # grid decrease
        loss=self.loss_func(q_value,target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # plot log
    def plot(self):
        plt.plot(self.log['episode'],self.log['success rate'])
        plt.ylabel("Success Rate")
        plt.xlabel("Episode")
        plt.title("N={}".format(self.env.len))
        plt.show()

    def update_target(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.target_network.close_grid()

    def save(self,file_path="./Module/module.npy"):
        torch.save(self.policy_network.state_dict(), file_path)

    # test current model for test_num times
    def test(self,test_num=100):
        success_rate = 0
        for k in range(test_num):
            success = self.eval()
            success_rate += success
            print("Test {}:".format(k+1))
            print("Target Squence: ",end="")
            print(self.env.target)
            print("Generated Squence: ",end="")
            print(self.env.state)
            if success!=0:
                print("Success!")
            else:
                print("Fail!")
        success_rate /= test_num
        print("Success Rate: {}".format(success_rate))

    # show the process of sequence generation with the target of <target>
    def show_process(self,target):
        self.policy_network.close_grid()
        self.policy_network.eval()
        obs = self.env.reset()
        self.env.target=np.array(target)
        done = False
        i=1
        while not done:
            action = self.select_action(0, obs['state'], obs['target'])
            print("Step {}:".format(i))
            i+=1
            print("Q-value of Two Actions: ",end="")
            print(self.policy_network(torch.cat([obs['state'], obs['target']],dim=-1).to(self.device)))
            obs, reward, done = self.env.step(action)
            print("Action: {}".format(action),end="  ")
            print("Reward: {}".format(reward))
            print("Target: ",end="")
            print(obs['target'])
            print("Current State: ",end="")
            print(obs['state'])
        if (obs['state']==obs['target']).all():
            print("Success!")
        else:
            print("Fail！")


class Agent_Curriculum():
    '''
    Input:
        env
        policy_network
        target_network
        replay_buffer
        gamma: discount factor
        lr: learning rate
        batch_size
        eps_start: start probability of ε-greedy
        eps_final: mininum probability of ε-greedy
        eps_decay: decay factor of ε-greedy
        device
        interval: curriculum interval
        initial_freeze: initial freezing proportion of parameters
        freeze_interval: number of curriculums to unfreeze parameters
    '''
    def __init__(self,env,policy_network,target_network=None,replay_buffer=None,gamma=0.99,lr=1e-3,batch_size=32,eps_start=1,eps_final=0.1,eps_decay=0.995,device=torch.device("cpu"),interval=1,initial_freeze=0.5,freeze_interval=3):
        self.env=env
        self.device=device
        self.policy_network=policy_network.to(self.device)
        self.target_network=target_network
        if self.target_network is not None:
            self.target_network=self.target_network.to(self.device)
            self.update_target()
        self.replay_buffer=replay_buffer
        self.gamma=gamma
        self.eps_start=eps_start
        self.eps_final=eps_final
        self.eps_decay=eps_decay
        self.eps=self.eps_start
        self.loss_func=nn.MSELoss()
        self.lr=lr
        self.optimizer=optim.Adam(self.policy_network.parameters(),lr=self.lr)
        self.batch_size=batch_size
        self.interval=interval
        self.freeze_interval=freeze_interval
        self.prob_interval=initial_freeze/(self.env.len/freeze_interval)
        self.prob=self.prob_interval*(self.env.len//freeze_interval)

    # select action by ε-greedy
    def select_action(self,eps,state,target):
        rand=torch.rand(1).item()
        if rand<eps: # choose action randomly
            return math.ceil(rand/(eps/(2**self.env.group)))
        else: # choose action with the max Q-value
            return self.policy_network(torch.cat([state,target],dim=-1).to(self.device)).max(dim=-1)[1].item()

    def update_policy(self):
        self.policy_network.open_grid()
        #sample batch
        state, target, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state=state.to(self.device)
        target=target.to(self.device)
        next_state=next_state.to(self.device)
        reward=reward.to(self.device)
        done=done.to(self.device)
        #compute real q_value and target q-value
        q_values=self.policy_network(torch.cat([state,target],dim=-1))
        q_value=torch.tensor([0.0]*len(action))
        for i in range(len(action)):
            q_value[i]=q_values[i,action[i]]
        next_q_value=self.target_network(torch.cat([next_state,target],dim=-1).to(self.device)).max(dim=-1)[0]
        target_q_value=reward+self.gamma*next_q_value * (1 - done)
        q_value=q_value.to(self.device)
        target_q_value=target_q_value.to(self.device)
        loss=self.loss_func(q_value,target_q_value)
        # grid decrease
        self.optimizer.zero_grad()
        loss.backward()
        self.policy_network.freeze(self.prob)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.target_network.close_grid()

    def save(self,file_path="./Module/module.npy"):
        torch.save(self.policy_network.state_dict(), file_path)

    '''
    Description: 
        evaluate current model
    Output:
        success or fail
    '''
    def eval(self,n):
        self.policy_network.close_grid()
        self.policy_network.eval()
        obs = self.env.reset(n)
        done = False
        while not done:
            action = self.select_action(0, obs['state'], obs['target'])
            obs, reward, done = self.env.step(action)
        return float((obs['state']==obs['target']).all())

    '''
    Description: 
        train policy network
    Input:
        episode: maximum episode
        update_episode: number of episodes to update target network
        save_path: path to save model parameters
        save_log: path to save log file
    '''
    def train(self,episode=50000,update_episode=10,save_path=None,save_log="./log"):
        self.log={"episode":[],"success rate":[],"fixed length":[],"current success rate":[],"current epoch":[]}
        # the fixed length will decrease steply until model convergences
        fixed_length=self.env.len-self.interval
        if fixed_length%self.freeze_interval==0:
            self.prob-=self.prob_interval
        index=1
        for i in range(1,int(episode)+1):
            loss=[]
            obs=self.env.reset(fixed_length)
            done=False
            # explore in environment
            while not done:
                self.policy_network.train()
                state=obs['state'].to(self.device)
                target=obs['target'].to(self.device)
                action=self.select_action(self.eps,state,target)
                obs,reward,done=self.env.step(action)
                self.replay_buffer.store({'state':state.to(self.device),'target':target.to(self.device),'reward':reward,'next_state':obs['state'].to(self.device),'action':action,'done':done})
            if index%5==0 and fixed_length!=0: # add some extra examples to avoid overfitting
                # obs = self.env.reset(0)
                obs = self.env.reset_state(np.random.randint(0,fixed_length))
                done = False
                while not done:
                    self.policy_network.train()
                    state = obs['state'].to(self.device)
                    target = obs['target'].to(self.device)
                    action = self.select_action(self.eps, state, target)
                    obs, reward, done = self.env.step(action)
                    self.replay_buffer.store(
                        {'state': state.to(self.device), 'target': target.to(self.device), 'reward': reward,
                         'next_state': obs['state'].to(self.device), 'action': action, 'done': done})
            # only when the buffer is full, the network will train
            if self.replay_buffer.is_full():
                loss.append(self.update_policy())
            if i % update_episode==0:
                self.update_target()
                # evaluate current model
                success_rate=0
                current_success_rate=0
                for k in range(100): # eval without fixed length
                    success_rate+=self.eval(0)
                success_rate/=100
                for k in range(100): # eval with fixed length
                    current_success_rate+=self.eval(fixed_length)
                current_success_rate/=100
                # write log & save model
                line1="Episode:{}  Success Rate:{}  Loss:{}".format(i,success_rate,np.average(loss))
                line2="Fixed Length:{}  Epoch:{}  Success Rate:{}".format(fixed_length,index,current_success_rate)
                print(line1)
                print(line2)
                if save_log is not None:
                    with open(save_log+".txt", 'a') as outfile:
                        outfile.write(line1+"\n")
                        outfile.write(line2+"\n")
                self.log["episode"].append(i)
                self.log["success rate"].append(success_rate)
                self.log["fixed length"].append(fixed_length)
                self.log["current success rate"].append(current_success_rate)
                self.log["current epoch"].append(index)
                index+=1
                loss.clear()
                if save_path is not None:
                    self.save(save_path)
                    print("Model Saved")
                # when the acc is larger than 0.9 for 5 continuous epoches, the fixed length will decrease
                if len(self.log["current epoch"])>5 and (np.array(self.log["current success rate"][-5:])>0.9).all():
                    if fixed_length==0:
                        break
                    fixed_length-=self.interval
                    if fixed_length<0:
                        fixed_length=0
                    else:
                        if fixed_length % self.freeze_interval == 0:
                            self.prob -= self.prob_interval
                        index = 1
            # ε decay
            if self.eps*self.eps_decay>self.eps_final:
                self.eps*=self.eps_decay
        np.save(save_log+".npy",self.log)

    # test current model for test_num times
    def test(self,test_num=100):
        success_rate = 0
        for k in range(test_num):
            success = self.eval(0)
            success_rate += success
            print("Test {}:".format(k+1))
            print("Target Squence: ",end="")
            print(self.env.target)
            print("Generated Squence: ",end="")
            print(self.env.state)
            if success!=0:
                print("Success!")
            else:
                print("Fail!")
        success_rate /= test_num
        print("Success Rate: {}".format(success_rate))

    # show the process of sequence generation with the target of <target>
    def show_process(self,target):
        self.policy_network.close_grid()
        self.policy_network.eval()
        obs = self.env.reset()
        self.env.target=np.array(target)
        done = False
        i=1
        while not done:
            action = self.select_action(0, obs['state'], obs['target'])
            print("Step {}:".format(i))
            i+=1
            print("Q-value of Two Actions: ",end="")
            print(self.policy_network(torch.cat([obs['state'], obs['target']],dim=-1).to(self.device)))
            obs, reward, done = self.env.step(action)
            print("Action: {}".format(action),end="  ")
            print("Reward: {}".format(reward))
            print("Target: ",end="")
            print(obs['target'])
            print("Current State: ",end="")
            print(obs['state'])
        if (obs['state']==obs['target']).all():
            print("Success!")
        else:
            print("Fail！")