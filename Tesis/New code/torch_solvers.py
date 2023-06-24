import torch
import numpy as np
from domains import *
from equations import *
import torch
import torch.nn as nn
from torch_architectures import *
from torch.utils.data import Dataset, DataLoader
import time
import pickle


from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

types=torch.float32
torch.set_default_dtype(types)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




def load_sol(config_file):
    dic=pickle.load(open(config_file, "rb"))
    domType=globals()[dic["dom_config"]["Domain"]]
    eqnType=globals()[dic["eqn_config"]["Equation"]]
    solType=globals()[dic["solver_config"]["Solver"]]
    dom=domType(dic["dom_config"])
    eqn=eqnType(dic["dom_config"],dic["eqn_config"])
    sol=solType(eqn,dic["solver_config"])
    sol.training_history=dic["training_history"]
    sol.currEps=dic["currEps"]
    return sol

class Solver():
    def __init__(self, eqn,solver_params):
        self.eqn=eqn
        self.solver_params=solver_params
        self.model=lambda x:0
        self.training_history=[]
        self.currEps=0
        self.dtype=solver_params["dtype"]

    def save_model(self,file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self,file_name):
        self.model.load_state_dict(torch.load(file_name))
    
    def save_sol(self,name):
        pickle.dump({"dom_config":self.eqn.domain.dom_config,
                     "eqn_config":self.eqn.eqn_config,
                     "solver_config":self.solver_params,
                     "training_history":self.training_history,
                     "currEps":self.currEps},open(name,'wb'))
        
    def control(self,t,X):
        pos=torch.tensor(np.hstack((t,X)),requires_grad=True)
        u=self.model(pos)
        V_x=torch.autograd.grad(u,pos)[0][1:]
        alpha=self.eqn.control(t,V_x)
        return alpha.detach().numpy()
    
    def plot_sol_static(self,t):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X,Y=self.eqn.domain.surface_plot_domain()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(-5,5)
        times=t*np.ones(np.ravel(X).shape[0])
        tes=torch.tensor(np.stack((times,np.ravel(X), np.ravel(Y)),axis=1),dtype=self.dtype)
        zs =self.model(tes).detach().numpy()
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z)
    
    def plot_solution(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X,Y=self.eqn.domain.surface_plot_domain()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(-5,5)

        fig.subplots_adjust(bottom=0.25)

        axfreq = fig.add_axes([0.25, 0.1, 0.5, 0.03])
        freq_slider = Slider(
            ax=axfreq,
            label='t',
            valmin=0.0,
            valmax=1.0,
            valinit=0.0,
            valstep=0.01
        )

        def draw(t):  
            ax.cla()
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set_zlim(-2,2)
            times=t*np.ones(np.ravel(X).shape[0])
            tes=torch.tensor(np.stack((times,np.ravel(X), np.ravel(Y)),axis=1),dtype=self.dtype)
            zs =self.model(tes).detach().numpy()
            Z = zs.reshape(X.shape)
            ax.plot_surface(X, Y, Z)
        draw(0)
        freq_slider.on_changed(draw)
        return freq_slider

class difussionSampleGeneratorBSDE(Dataset):
    def __init__(self, eqn, num_samples,dt,Ndis,in_region,test_point):
        self.eqn=eqn
        self.num_samples=num_samples
        self.dt=dt
        self.Ndis=Ndis
        self.in_region=in_region
        self.test_point=test_point
        if in_region:
            self.samples=self.eqn.simulate_N_interior_diffusion(dt,Ndis,num_samples)
        else:
            self.samples=self.eqn.simulate_N_interior_diffusion(dt,Ndis,num_samples,X0=self.test_point)
    
    def re_sample(self):
        if self.in_region:
            self.samples=self.eqn.simulate_N_interior_diffusion(self.dt,self.Ndis,self.num_samples)
        else:
            self.samples=self.eqn.simulate_N_interior_diffusion(self.dt,self.Ndis,self.num_samples,X0=self.test_point)

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.samples

class Deep_BSDE_Solver(Solver):
    """The fully connected neural network model."""
    def __init__(self,eqn, solver_params):
        self.solver_params=solver_params
        self.solver_params["Solver"]='Deep_BSDE_Solver'
        self.eqn = eqn
        self.Ndis=self.solver_params["Ndis"]
        self.dt=self.eqn.domain.total_time/(self.Ndis-1)
        self.solver_params["net_config"]["Ndis"]=self.Ndis
        self.solver_params["net_config"]["dt"]=self.dt
        self.net_config = self.solver_params["net_config"]
        self.net_config["in_region"]=self.solver_params["in_region"]
        self.in_region=self.solver_params["in_region"]
        if self.net_config["net_type"]=='Normal':
            self.model = Global_Model_Deep_BSDE(self.net_config, eqn).to(device)
        if self.net_config["net_type"]=='Merged':
            self.model = Global_Model_Merged_Deep_BSDE(self.net_config, eqn).to(device)
        #self.model=torch.compile(self.model)
        self.initial_lr=solver_params["initial_lr"]
        self.sample_every=solver_params["sample_every"]
        self.Nsamp=solver_params["Nsamp"]
        self.logging_interval=solver_params["logging_interval"]
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.initial_lr,weight_decay=0.00001)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,self.lr_schedule)
        self.test_point=torch.ones(eqn.dim)*0.9
        self.dataGenerator=difussionSampleGeneratorBSDE(eqn, self.Nsamp,self.dt,self.Ndis,self.in_region,self.test_point)  
        self.true_sol=eqn.true_solution(0.0,self.test_point,10000,0.001).numpy()
        print(self.true_sol)
        self.training_history=[]
        self.currEps=0

    def lr_schedule(self,epoch):
        #return 10.0/(epoch+1)
        return 1.0
    
    def plot_solution(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X,Y=self.eqn.domain.surface_plot_domain()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(-5,5)

        fig.subplots_adjust(bottom=0.25)

        axfreq = fig.add_axes([0.25, 0.1, 0.5, 0.03])
        freq_slider = Slider(
            ax=axfreq,
            label='t',
            valmin=0.0,
            valmax=1.0,
            valinit=0.0,
            valstep=0.01
        )

        def draw(t):  
            ax.cla()
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set_zlim(-2,2)
            times=t*np.ones(np.ravel(X).shape[0])
            tes=torch.tensor(np.stack((np.ravel(X), np.ravel(Y)),axis=1),dtype=self.dtype)
            zs =self.model.y0(tes).detach().numpy()
            Z = zs.reshape(X.shape)
            ax.plot_surface(X, Y, Z)
        draw(0)
        freq_slider.on_changed(draw)
        return freq_slider
    
    def simulate_trajectory(self,t0,X0):
        self.model.eval()
        return self.eqn.simulate_controlled_trajectory(t0,X0,self.Ndis,self.dt,self.model.grad_func,plot=True)
    
    def train(self,steps):
        start_time = time.time()
        valid_data = self.dataGenerator[0]

        data_loader=DataLoader(self.dataGenerator, 
                                        batch_size=None,
                                        shuffle=False,
                                        num_workers=0)

        # begin sgd iteration
        for step in range(steps+1):
            #print(step)
            
            if (step)%self.sample_every==0:
                self.dataGenerator.re_sample()
                data_loader=DataLoader(self.dataGenerator, 
                                        batch_size=None,
                                        shuffle=False,
                                        num_workers=0)
            inputs=self.dataGenerator[0]
            results=self.model(inputs)
            loss=self.loss_fn(inputs,results)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            if self.currEps % self.logging_interval==0:
                #loss = self.loss_fn(valid_data, self.model(valid_data)).detach().numpy()
                y_init = self.model.evaluate_y_0(self.test_point).detach().numpy()[0]
                elapsed_time = time.time() - start_time
                err=np.abs(y_init-self.true_sol)
                #self.training_history.append([step, loss, y_init, err, elapsed_time])
                self.training_history.append([step, y_init, err, elapsed_time])
                #print("Epoch ",self.currEps, " y_0 ",y_init," time ", elapsed_time," loss ", loss, " error ",err)
                print("Epoch ",self.currEps, " y_0 ",y_init," time ", elapsed_time, " error ",err)
            self.currEps+=1

        return np.array(self.training_history)

    def loss_fn(self, inputs,results):
        DELTA_CLIP = 50.0
        dw, x = inputs
        delta = results - self.eqn.g_tf(x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP, torch.square(delta),
                                       2 * DELTA_CLIP * torch.abs(delta) - DELTA_CLIP ** 2))

        return loss
    