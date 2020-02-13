from pdb import set_trace as T
from tqdm import tqdm
import torch
import numpy as np
import gym
import time

def collectData(env, episodes=400):
   '''Collect (s, a, r, s_next) experience with a 0-1 reward'''
   obs, atns, rewards, nxtObs = [], [], [], []
   for i in tqdm(range(episodes), desc='Data Generation'):
      nxtOb, done = env.reset(), False
      while not done:
         ob   = nxtOb
         atn  = env.action_space.sample()
         nxtOb, reward, done, _ = env.step(atn)

         if done:
            reward = -1
         else:
            reward = 0

         obs.append(ob)
         atns.append(atn)
         rewards.append(reward)
         nxtObs.append(nxtOb)

   return obs, atns, rewards, nxtObs

class Dataset:
   '''Experience shuffler'''
   def __init__(self, obs, atns, rewards, nxtObs):
      assert len(obs) == len(atns) == len(rewards) == len(nxtObs)
      self.n = len(obs)

      self.obs     = torch.tensor(obs).float()
      self.atns    = torch.tensor(atns).float()
      self.rewards = torch.tensor(rewards).float()
      self.nxtObs  = torch.tensor(nxtObs).float()

   def shuffled(self):
      idxs  = np.arange(self.n)
      idxs  = np.random.permutation(idxs)

      return (self.obs[idxs], self.atns[idxs],
            self.rewards[idxs], self.nxtObs[idxs])

class Policy(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.fc1 = torch.nn.Linear(4, 16)
      self.fc2 = torch.nn.Linear(16, 2)

   def forward(self, x):
      x = torch.relu(self.fc1(x)) 
      x = self.fc2(x)
      return x

def cost(Q, s, a, r, sn, gamma=0.95):
   '''Squared error Qfn loss'''
   Qs  = Q(s)[range(len(a)), a.long()]
   Qsn = torch.max(Q(sn), 1)[0].detach()
   return (r + gamma*Qsn - Qs)**2 

def NFQ_main(D, epochs=100):  
   Q = Policy()
   optim = torch.optim.Adam(Q.parameters())
   desc = '[Loss={}] Batch Training'
   bar = tqdm(range(epochs), desc=desc.format('N/A'))
   for i in bar:
      obs, atns, rewards, nxtObs = D.shuffled()
      err = cost(Q, obs, atns, rewards, nxtObs).mean()
      bar.set_description(desc.format('{:.2f}'.format(err)))

      err.backward()
      optim.step()

   return Q
      
def test(env, policy):
   '''Render the policy'''
   ob = env.reset()
   t = 0
   while True:
      ob  = torch.tensor(ob).float()
      p = policy(ob)
      atn = int(torch.argmax(p))
      ob, r, done, _  = env.step(atn)

      time.sleep(0.1)
      env.render()
      if done:
         env.reset()

         print(t)
         t = 0
      t += 1
       
if __name__ == '__main__':
   train = True 
   env   = gym.make('CartPole-v0')

   '''Algorithm is unstable -- expect
   good performance at most 1/5 runs'''
   if train:
      data = collectData(env)
      D    = Dataset(*data)
      policy = NFQ_main(D)
      torch.save(policy, 'policy.pth')
   else:
      policy = torch.load('policy.pth')

   test(env, policy)
