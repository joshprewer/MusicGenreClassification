import numpy as np
import random
from itertools import combinations as cb
import math
from copy import deepcopy as dc
from tqdm import tqdm
from sklearn import utils, model_selection

"""Using Algorithm
* Binary Cuckoo Search
* Binary Dragon Fly Algorithm
"""

"""Evaluate Function """
class Evaluate:
    def __init__(self, input_X, input_Y, clf):
        self.input_X = input_X
        self.input_Y = input_Y
        self.clf = clf
        
    def evaluate(self,gen):
        feature_sets = np.nonzero(gen)[0]

        if feature_sets.size == 0:
            return 0
        else:
            x = np.take(self.input_X, feature_sets, axis=1)
            xs, ys = utils.shuffle(x, self.input_Y)

            cv = len(np.unique(ys))
            cv_results = model_selection.cross_val_score(self.clf, xs, ys, cv=cv)            
        
            return cv_results.mean()
                
        
    def check_dimentions(self,dim):#check number of all feature
        if dim==None:
            return len(self.input_X[0])
        else:
            return dim

"""Common Function"""
def random_search(n,dim):
    """
    create genes list
    input:{ n: Number of population, default=20
            dim: Number of dimension
    }
    output:{genes_list → [[0,0,0,1,1,0,1,...]...n]
    }
    """
    gens=[[0 for g in range(dim)] for _ in range(n)]
    for i,gen in enumerate(gens) :
        r=random.randint(1,dim)
        for _r in range(r):
            gen[_r]=1
        random.shuffle(gen)
    return gens

"""BCS"""
def sigmoid(x):
    try:
        return 1/(1+math.exp(-x))
    except OverflowError:
        return 0.000001
def sigma(beta):
    p=math.gamma(1+beta)* math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*(pow(2,(beta-1)/2)))
    return pow(p,1/beta)
def levy_flight(beta,best,est,alpha):
    sg=sigma(beta)
    u=np.random.normal(0,sg**2)
    v=abs(np.random.normal(0,1))
    step=u/pow(v,1/beta)
    step_size=alpha+step#+(step*(est-best))
    new=est+step_size#*np.random.normal()#random.normalvariate(0,sg)
    return new

def BCS(Eval_Func,m_i=200,n=20,minf=0,dim=None,prog=False,alpha=0.1,beta=1.5,param=0.25):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            alpha and beta: Arguments in levy flight, default=0.1,1.5
            param: Probability to destroy inferior nest, default=0.25(25%)
            }

    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func.evaluate
    if dim==None:
        dim=Eval_Func.check_dimentions(dim)
    pa=param
    #flag=dr
    gens=random_search(n,dim)
    fit=[float("-inf") if minf == 0 else float("inf") for _ in range(n)]
    pos=[0 for _ in range(n)]
    g_pos=[0]*dim
    g_val=float("-inf") if minf == 0 else float("inf")
    gens_dict={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)
    for it in miter:
        for i,g in enumerate(gens):
            if tuple(g) in gens_dict:
                score=gens_dict[tuple(g)]
            else:
                score=estimate(g)
                gens_dict[tuple(g)]=score
            if score > fit[i] if minf==0 else score < fit[i]:
                fit[i]=score
                pos[i]=g

        maxfit,maxind=max(fit),fit.index(max(fit))
        minfit,minind=min(fit),fit.index(min(fit))
        if minf==0:
            if maxfit > g_val:
                g_val=dc(maxfit)
                g_pos=dc(gens[maxind])
        else:
            if minfit < g_val:
                g_val=dc(minfit)
                g_pos=dc(gens[minind])

        if pa < random.uniform(0,1):
            if minf==0:
                gens[minind]=[0 if 0.5>random.uniform(0,1) else 1 for _ in range(dim)]#rand_gen()
                fit[minind]=float("-inf") if minf == 0 else float("inf")
            else:
                gens[maxind]=[0 if 0.5>random.uniform(0,1) else 1 for _ in range(dim)]#rand_gen()
                fit[maxind]=float("-inf") if minf == 0 else float("inf")


        for g in gens:
            for d in range(dim):
                x=levy_flight(beta,g_pos[d],g[d],alpha)
                if random.uniform(0,1) < sigmoid(x):
                    g[d]=1
                else:
                    g[d]=0
        
        if g_val == 1.0:
            break
                    
    return g_val,g_pos,g_pos.count(1)

"""BDFA"""
def BDFA(Eval_Func,n=20,m_i=200,dim=None,minf=0,prog=False):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            }

    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func.evaluate
    if dim==None:
        dim=Eval_Func.check_dimentions(dim)
    maxiter=m_i#500
    #flag=dr#True
    best_v=float("-inf") if minf == 0 else float("inf")
    best_p=[0]*dim
    gens_dict={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}

    enemy_fit=float("-inf") if minf == 0 else float("inf")
    enemy_pos=[0 for _ in range(dim)]
    food_fit=float("-inf") if minf == 0 else float("inf")
    food_pos=[0 for _ in range(dim)]

    fit=[0 for _ in range(n)]
    genes=random_search(n,dim)
    genesX=random_search(n,dim)

    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)
    for it in miter:
        w=0.9 - it * ((0.9-0.4) / maxiter)
        mc=0.1- it * ((0.1-0) / (maxiter/2))
        if mc < 0:
            mc=0
        s=2 * random.random() * mc
        a=2 * random.random() * mc
        c=2 * random.random() * mc
        f=2 * random.random()
        e=mc
        if it > (3*maxiter/3):
            e=0

        for i in range(n):
            if tuple(genes[i]) in gens_dict:
                fit[i]=gens_dict[tuple(genes[i])]
            else:
                fit[i]=estimate(genes[i])
                gens_dict[tuple(genes[i])]=dc(fit[i])
            if fit[i] > food_fit if minf==0 else fit[i] < food_fit:
                food_fit=dc(fit[i])
                food_pos=dc(genes[i])

            if fit[i] > enemy_fit if minf==0 else fit[i] < enemy_fit:
                enemy_fit=dc(fit[i])
                enemy_pos=dc(genes[i])

        for i in range(n):
            ind=-1
            nn=-1
            ndx=[[0 for _d in range(dim)] for _ in range(n)]
            nx=[[0 for _d in range(dim)] for _ in range(n)]

            for j in range(n):
                if i==j:
                    pass
                else:
                    ind+=1
                    nn+=1
                    ndx[ind]=dc(genesX[j])
                    nx[ind]=dc(genes[j])

            S=[0 for _ in range(dim)]
            for k in range(nn):
                S=[_s+(_x-_y) for _s,(_x,_y) in zip(S,zip(ndx[k],genes[i]))] #s+(nx[k]-x[i])
            S=S

            A=[sum([_[_d] for _ in ndx])/nn for _d in range(dim)]#[sum(_)/nn if _ != 0 else 0 for _ in ndx]
            #[_-g for _,g in zip([sum([_[_d] for _ in nx])/nn for _d in range(dim)],genes[i])]
            C=[_-g for _,g in zip([sum([_[_d] for _ in nx])/nn for _d in range(dim)],genes[i])]#[sum(_)/nn-g if _ != 0 else 0 for _,g in zip(nx,genes[i])]

            F=[fp-g for fp,g in zip(food_pos,genes[i])]
            E=[ep+g for  ep,g in zip(enemy_pos,genes[i])]

            for j in range(dim):
                genesX[i][j]=s*S[j]+a*A[j]+c*C[j]+ f *F[j]+e*E[j]+w*genesX[i][j]

                if genesX[i][j] > 6:
                    genesX[i][j]=6
                if genesX[i][j] < -6:
                    genesX[i][j]=-6
                T = abs(genesX[i][j] / math.sqrt((1+genesX[i][j]**2)))
                if random.random()<T:
                    genes[i][j]=1 if genes[i][j] == 0 else 0
    best_p=food_pos
    best_v=food_fit

    return best_v,best_p,best_p.count(1)
