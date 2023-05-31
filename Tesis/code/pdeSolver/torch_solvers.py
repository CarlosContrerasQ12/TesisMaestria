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



class DGM_solver(Solver):
    """
    Class for training a neural network for approximating a solution to a PDE
    using the DGM algorithm.
    """
    def __init__(self, eqn,solver_params):
        super(DGM_solver,self).__init__(eqn,solver_params)
        self.solver_params["Solver"]='DGM_solver'
        self.initial_lr=solver_params["initial_lr"]
        self.net_structure=globals()[solver_params["net_structure"]]
        self.model=self.net_structure(solver_params["net_config"],eqn)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.initial_lr,weight_decay=0.00001)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,self.lr_schedule)
        self.Lweights=solver_params["initial_loss_weigths"]
        self.logging_interval=solver_params["logging_interval"]
        self.Nsamp=solver_params["N_samples_per_batch"]
        self.sample_every=solver_params["sample_every"]
        torch.set_default_dtype(self.dtype)
        self.currEps=0
        self.interior_valid=torch.as_tensor(self.eqn.interior_point_sample(2048),dtype=self.dtype).requires_grad_()
        np_neu,np_n_neu=self.eqn.neumann_sample(2048,0)
        self.neumann_valid=(torch.as_tensor(np_neu,dtype=self.dtype).requires_grad_(),
                            torch.as_tensor(np_n_neu,dtype=self.dtype).requires_grad_())
        self.dirichlet_valid=torch.as_tensor(self.eqn.dirichlet_sample(2048,0),dtype=self.dtype).requires_grad_()
        self.terminal_valid=torch.as_tensor(self.eqn.terminal_sample(2048),dtype=self.dtype).requires_grad_()

    def lr_schedule(self,epoch):
        return 10.0/(epoch+1)

    def set_lr(self,lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def loss(self,interior_sample,neumann_sample,dirichlet_sample,terminal_sample):

        tLi=time.time()
        V=self.model(interior_sample)
        dV=torch.autograd.grad(V,
                           interior_sample,
                           grad_outputs=torch.ones_like(V),
                           retain_graph=True,
                           create_graph=True,
                           only_inputs=True)[0]
        V_t=dV[:,0]
        V_x=dV[:,1:]
        V_xxn=torch.autograd.grad(dV,
                        interior_sample,
                        grad_outputs=torch.ones_like(dV),
                        retain_graph=True,
                        create_graph=False,
                        only_inputs=True)[0][:,1:]
        nV_x2=torch.sum(V_x*V_x,axis=1)
        V_xx=torch.sum(V_xxn,axis=1)
        diff_V=self.eqn.Nv(interior_sample, V,V_t,nV_x2,V_xx)
        Li=torch.mean(torch.square(diff_V))
        print("Li:",time.time()-tLi)

        tb=time.time()
        x_neumann,n_neumann=neumann_sample
        Vn=self.model(x_neumann)
        dVn=torch.autograd.grad(Vn,
                                x_neumann,
                                grad_outputs=torch.ones_like(Vn),
                                retain_graph=True,
                                create_graph=False,
                                only_inputs=True)[0]
        V_nx=dVn[:,1:]
        normaldVn=torch.sum(V_nx*n_neumann,axis=1)
        Ln=torch.mean(torch.square(normaldVn-self.eqn.h_n(x_neumann)))
        
        Vd=self.model(dirichlet_sample)
        Ld=torch.mean(torch.square(Vd-self.eqn.h_d(dirichlet_sample)))
        
        Vter=self.model(terminal_sample)
        LT=torch.mean(torch.square(Vter-self.eqn.g_Tf(terminal_sample)))
        print("Lb ",time.time()-tb)

        return Li,Ln,Ld,LT
    
    def train(self,epochs):
        
        training_history = []
        print('Comencemos a calcular')
        start_time=time.time()
        interior = torch.as_tensor(self.eqn.interior_point_sample(self.Nsamp),dtype=self.dtype).requires_grad_()
        np_neumann,np_n_neumann=self.eqn.neumann_sample(self.Nsamp,0)
        neumann= (torch.as_tensor(np_neumann,dtype=self.dtype).requires_grad_(),
                torch.as_tensor(np_n_neumann,dtype=self.dtype).requires_grad_())
        dirichlet=torch.as_tensor(self.eqn.dirichlet_sample(self.Nsamp,0),dtype=self.dtype).requires_grad_()
        terminal=torch.as_tensor(self.eqn.terminal_sample(self.Nsamp),dtype=self.dtype).requires_grad_()

        # Begin iteration over samples
        for step in range(epochs):
            ti=time.time()
            if (step+1)%self.sample_every==0:
                interior = torch.as_tensor(self.eqn.interior_point_sample(self.Nsamp),dtype=self.dtype).requires_grad_()
                np_neumann,np_n_neumann=self.eqn.neumann_sample(self.Nsamp,0)
                neumann= (torch.as_tensor(np_neumann,dtype=self.dtype).requires_grad_(),
                        torch.as_tensor(np_n_neumann,dtype=self.dtype).requires_grad_())
                dirichlet=torch.as_tensor(self.eqn.dirichlet_sample(self.Nsamp,0),dtype=self.dtype).requires_grad_()
                terminal=torch.as_tensor(self.eqn.terminal_sample(self.Nsamp),dtype=self.dtype).requires_grad_()
            
            
            Li,Ln,Ld,LT=self.loss(interior,neumann,dirichlet,terminal)
            L=(self.Lweights[0]*Li)+(self.Lweights[1]*Ln)+(self.Lweights[2]*Ld)+(self.Lweights[3]*LT)
            self.optimizer.zero_grad(set_to_none=True)
            L.backward()
            self.optimizer.step()
            self.scheduler.step()
            print("Total time:",time.time()-ti)

            if step % self.logging_interval==0:
                L1,L2,L3,L4 = self.loss(self.interior_valid,
                                        self.neumann_valid,
                                        self.dirichlet_valid,
                                        self.terminal_valid)
                L=(self.Lweights[0]*Li)+(self.Lweights[1]*Ln)+(self.Lweights[2]*Ld)+(self.Lweights[3]*LT)
                loss=L.detach().numpy()
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, elapsed_time])
                print("Epoch ",self.currEps,
                      " time ", elapsed_time,
                      " loss ", loss, 
                      " L1:",L1.item(),
                      " L2:",L2.item(),
                      " L3: ",L3.item(),
                      " L4: ",L4.item())
            self.currEps+=1
        return np.array(training_history)
    
class difussionSampleGenerator(Dataset):
    def __init__(self, eqn, num_samples,solver_config):
        self.eqn=eqn
        self.N_samples=num_samples
        self.solver_config=solver_config
        self.dt=solver_config["dt"]
        self.N_max=solver_config["N_max"]
        self.dtype=solver_config["dtype"]
        samp=self.eqn.interior_point_sample(self.N_samples)
        t0=samp[:,0]
        X0=samp[:,1:]
        salp=self.eqn.simulate_N_interior_diffusion(self.dt,self.N_max,t0,X0,self.N_samples)
        self.samples=[torch.as_tensor(path,dtype=self.dtype) for path in salp]

    def re_sample(self):
        samp=self.eqn.interior_point_sample(self.N_samples)
        t0=samp[:,0]
        X0=samp[:,1:]
        salp=self.eqn.simulate_N_interior_diffusion(self.dt,self.N_max,t0,X0,self.N_samples)
        self.samples=[torch.as_tensor(path,dtype=self.dtype) for path in salp]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class Interp_PINN_BSDE_solver(Solver):
    """
    Class for training a neural network for approximating a solution to a PDE
    using the interpolation between PINNS and BSDE algorithm.
    """
    def __init__(self, eqn,solver_params):
        super(Interp_PINN_BSDE_solver,self).__init__(eqn,solver_params)
        self.solver_params=solver_params
        self.solver_params["Solver"]='Interp_PINN_BSDE_solver'
        self.initial_lr=solver_params["initial_lr"]
        self.net_structure=solver_params["net_size"]
        self.net_structure=globals()[solver_params["net_structure"]]
        self.model=self.net_structure(solver_params["net_config"],eqn)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.initial_lr,weight_decay=0.00001)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,self.lr_schedule)
        self.Lweights=solver_params["initial_loss_weigths"]
        self.logging_interval=solver_params["logging_interval"]
        self.Nsamp_interior=solver_params["N_samples_per_batch_interior"]
        self.Nsamp_boundary=solver_params["N_samples_per_batch_boundary"]
        self.sample_every=solver_params["sample_every"]
        self.dt=solver_params["dt"]
        self.N_max=solver_params["N_max"]
        self.alpha=solver_params["alpha"]
        self.adaptive_weigths=solver_params["adaptive_weigths"]
        torch.set_default_dtype(self.dtype)
        self.currEps=0
        self.dataGenerator=difussionSampleGenerator(eqn, self.Nsamp_interior,solver_params)
        #self.interior_valid=torch.as_tensor(self.dataGenerator[0],dtype=self.dtype).requires_grad_()
        np_neu,np_n_neu=self.eqn.neumann_sample(2048,0)
        self.neumann_valid=(torch.as_tensor(np_neu,dtype=self.dtype).requires_grad_(),
                            torch.as_tensor(np_n_neu,dtype=self.dtype).requires_grad_())
        self.dirichlet_valid=torch.as_tensor(self.eqn.dirichlet_sample(2048,0),dtype=self.dtype).requires_grad_()
        self.terminal_valid=torch.as_tensor(self.eqn.terminal_sample(2048),dtype=self.dtype).requires_grad_()

    def lr_schedule(self,epoch):
        return 10.0/(epoch+1)

    def set_lr(self,lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def plot_N_agent_sample_path(self):
        X=self.dataGenerator[0]
        self.eqn.domain.plot_N_agent_sample_path(X)

    def sample_paths_loss(self,path):
        X=path[0,:]
        Xis=path[1,:-1,1:]
        dtf=X[-1,0]-X[-2,0]
        V=self.model(X.requires_grad_())
        dV=torch.autograd.grad(V,X,
                                grad_outputs=torch.ones_like(V),
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]
        V_x=dV[:,1:]
        term1=np.sqrt(self.dt)*torch.sum(V_x[:-2]*Xis[:-1])+torch.sum(V_x[-2]*Xis[-1])*np.sqrt(dtf)
        nV_x2=torch.nansum(V_x*V_x,axis=1)
        term2=self.dt*(torch.sum(self.eqn.f(X[:-2,1:],V[:-2],nV_x2[:-2])))
        term3=dtf*self.dt*(self.eqn.f(X[-2,1:],V[-2],nV_x2[-2]))
        #dV.requires_grad_(False)
        #V.requires_grad_(False)
        X.requires_grad_(False)
        return torch.square(V[-1]-V[0]-term1+term2+term3)

    def interior_loss(self,dataLoader):
        err=0.0
        losses=[self.sample_paths_loss(path) for path in dataLoader]
        for l in losses:
            err+=l
        return err/self.Nsamp_interior

    def loss(self,dirichlet_data_loader,neumann_sample,dirichlet_sample,terminal_sample):
        tii=time.time()
        Li=self.interior_loss(dirichlet_data_loader)
        print("Li:",time.time()-tii)

        tb=time.time()
        x_neumann,n_neumann=neumann_sample
        Vn=self.model(x_neumann)
        dVn=torch.autograd.grad(Vn,x_neumann,
                                grad_outputs=torch.ones_like(Vn),
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]
        V_nx=dVn[:,1:]
        normaldVn=torch.sum(V_nx*n_neumann,axis=1)
        Ln=torch.mean(torch.square(normaldVn-self.eqn.h_n(x_neumann)))
        
        Vd=self.model(dirichlet_sample)
        Ld=torch.mean(torch.square(Vd-self.eqn.h_d(dirichlet_sample)))
        
        Vter=self.model(terminal_sample)
        LT=torch.mean(torch.square(Vter-self.eqn.g_Tf(terminal_sample)))
        print("Lb:",time.time()-tb)
        return Li,Ln,Ld,LT
    
    def train(self,epochs):
        
        training_history = []
        print('Comencemos a calcular')
        start_time=time.time()
        dirichlet_data_loader=DataLoader(self.dataGenerator, 
                                        batch_size=None,
                                        shuffle=False,
                                        num_workers=0)
        np_neumann,np_n_neumann=self.eqn.neumann_sample(self.Nsamp_boundary,0)
        neumann= (torch.as_tensor(np_neumann,dtype=self.dtype).requires_grad_(),
                torch.as_tensor(np_n_neumann,dtype=self.dtype).requires_grad_())
        dirichlet=torch.as_tensor(self.eqn.dirichlet_sample(self.Nsamp_boundary,0),dtype=self.dtype).requires_grad_()
        terminal=torch.as_tensor(self.eqn.terminal_sample(self.Nsamp_boundary),dtype=self.dtype).requires_grad_()

        # Begin iteration over samples
        for step in range(1,epochs+1):
            self.currEps+=1
            print(step)
            tis=time.time()
            if (step)%self.sample_every==0:
                self.dataGenerator.re_sample()
                dirichlet_data_loader=DataLoader(self.dataGenerator, 
                                        batch_size=None,
                                        shuffle=False,
                                        num_workers=0)
                np_neumann,np_n_neumann=self.eqn.neumann_sample(self.Nsamp_boundary,0)
                neumann= (torch.as_tensor(np_neumann,dtype=self.dtype).requires_grad_(),
                        torch.as_tensor(np_n_neumann,dtype=self.dtype).requires_grad_())
                dirichlet=torch.as_tensor(self.eqn.dirichlet_sample(self.Nsamp_boundary,0),dtype=self.dtype).requires_grad_()
                terminal=torch.as_tensor(self.eqn.terminal_sample(self.Nsamp_boundary),dtype=self.dtype).requires_grad_()
            Li,Ln,Ld,LT=self.loss(dirichlet_data_loader,neumann,dirichlet,terminal)
            self.optimizer.zero_grad(set_to_none=True)
            if self.adaptive_weigths:
                dLi=torch.autograd.grad(outputs=Li,inputs=self.model.parameters(),retain_graph=True)
                dLn=torch.autograd.grad(outputs=Ln,inputs=self.model.parameters(),retain_graph=True,allow_unused=True)
                dLd=torch.autograd.grad(outputs=Ld,inputs=self.model.parameters(),retain_graph=True)
                dLT=torch.autograd.grad(outputs=LT,inputs=self.model.parameters(),retain_graph=True)
                maxLi=0.0
                sumLn=0.0
                sumLd=0.0
                sumLT=0.0
                tamLn=0
                tamLd=0
                tamLT=0

                for i, w in enumerate(self.model.parameters()):
                    gdL=[dLi[i],dLn[i],dLd[i],dLT[i]]
                    if dLi[i]==None: gdL[0]=torch.Tensor([0.0])
                    if dLn[i]==None: gdL[1]=torch.Tensor([0.0])
                    if dLd[i]==None: gdL[2]=torch.Tensor([0.0])
                    if dLT[i]==None: gdL[3]=torch.Tensor([0.0]) 
                    dL=(self.Lweights[0]*gdL[0])+(self.Lweights[1]*gdL[1])+(self.Lweights[2]*gdL[2])+(self.Lweights[3]*gdL[3])
                    #print(i,w.grad[0],dL[0])
                    w.grad=dL

                    gLiMax=torch.max(torch.abs(gdL[0])).detach().numpy()
                    sumLn+=torch.sum(torch.abs(gdL[1])).detach().numpy()
                    sumLd+=torch.sum(torch.abs(gdL[2])).detach().numpy()
                    sumLT+=torch.sum(torch.abs(gdL[3])).detach().numpy()
                    if gLiMax>maxLi: maxLi=gLiMax
                    tamLn+=gdL[1].shape[0]
                    tamLd+=gdL[2].shape[0]
                    tamLT+=gdL[3].shape[0]
                lamhat=[1.0,maxLi*tamLn/sumLn,maxLi*tamLd/sumLd,maxLi*tamLd/sumLd]
                lamnew=[1.0 if i==0 else (1-self.alpha)*self.Lweights[i]+self.alpha*lamhat[i] for i in range(4)]
                self.Lweights=lamnew
                print("Nuevos Lamn",self.Lweights)
                self.optimizer.step()
                self.scheduler.step()
            else:
                L=(self.Lweights[0]*Li)+(self.Lweights[1]*Ln)+(self.Lweights[2]*Ld)+(self.Lweights[3]*LT)
                L.backward()
                self.optimizer.step()
                self.scheduler.step()

            print("Tiempto toal",time.time()-tis)

            if step % self.logging_interval==0:
                L1,L2,L3,L4 = self.loss(dirichlet_data_loader,
                                        self.neumann_valid,
                                        self.dirichlet_valid,
                                        self.terminal_valid)
                L=(self.Lweights[0]*Li)+(self.Lweights[1]*Ln)+(self.Lweights[2]*Ld)+(self.Lweights[3]*LT)
                loss=L.detach().numpy()
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, elapsed_time])
                print("Epoch ",self.currEps,
                      " time ", elapsed_time,
                      " loss ", loss, 
                      " L1:",L1.item(),
                      " L2:",L2.item(),
                      " L3: ",L3.item(),
                      " L4: ",L4.item())
            
        return np.array(training_history)


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
        if self.net_config["net_type"]=='Merged_residual':
            self.model = Global_Model_Merged_Residual_Deep_BSDE(self.net_config, eqn).to(device)
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
    

class Raissi_BSDE_Solver(object):
    def __init__(self,eqn, solver_params):
        
        self.solver_params=solver_params
        self.solver_params["Solver"]='Raissi_BSDE_Solver'
        self.eqn = eqn
        self.Ndis=self.solver_params["Ndis"]
        self.dt=self.eqn.domain.total_time/(self.Ndis-1)
        self.times=np.arange(0, self.Ndis) * self.dt
        self.net_config = self.solver_params["net_config"]
        self.model=Global_Raissi_Net(self.net_config,eqn)
        self.initial_lr=solver_params["initial_lr"]
        self.sample_every=solver_params["sample_every"]
        self.Nsamp=solver_params["Nsamp"]
        self.logging_interval=solver_params["logging_interval"]
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.initial_lr,weight_decay=0.00001)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,self.lr_schedule)
        self.test_point=torch.ones(eqn.dim)*0.9
        self.in_region=False
        self.dataGenerator=difussionSampleGeneratorBSDE(eqn, self.Nsamp,self.dt,self.Ndis,self.in_region,self.test_point)  
        self.true_sol=eqn.true_solution(0.0,self.test_point,50000,0.01).numpy()
        print(self.true_sol)
        self.training_history=[]
        self.currEps=0

    def lr_schedule(self,epoch):
        #return 10.0/(epoch+1)
        return 1.0
    
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
            loss=self.loss_fn2(inputs)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            if self.currEps % self.logging_interval==0:
                #loss = self.loss_fn(valid_data).detach().numpy()
                tx=torch.hstack((torch.zeros(1),self.test_point))
                y_init = self.model(tx).detach().numpy()[0]
                elapsed_time = time.time() - start_time
                err=np.abs(y_init-self.true_sol)
                #self.training_history.append([step, loss, y_init, err, elapsed_time])
                #print("Epoch ",self.currEps, " y_0 ",y_init," time ", elapsed_time," loss ", loss, " error ",err)
                self.training_history.append([step, y_init, err, elapsed_time])
                print("Epoch ",self.currEps, " y_0 ",y_init," time ", elapsed_time, " error ",err)
            self.currEps+=1

        return np.array(self.training_history)

    def Dg_tf(self,X): # M x D
        Gt=self.eqn.g_tf(X)
        return torch.autograd.grad(Gt, X, create_graph=True,grad_outputs=torch.ones_like(Gt),allow_unused=True)[0]
    
    
    def loss_fn(self, inputs):
        loss=0.0
        dw, x = inputs
        tx=torch.hstack((self.times[0]*torch.ones((x.shape[0], 1)),x[:,:,0]))
        tx.requires_grad_()
        Y0=self.model(tx)
        Z0 = torch.autograd.grad(Y0, tx, create_graph=True,grad_outputs=torch.ones_like(Y0),allow_unused=True)[0][:,1:]
        
        for t in range(self.Ndis-1):
            print(self.eqn.f_tf(self.times[t],x[:,:,0],Y0,Z0).shape)
            print((Z0 * dw[:, :, t]).shape)
            Y1_tilde = Y0 + self.eqn.f_tf(self.times[t],x[:,:,0],Y0,Z0)*self.dt + torch.sum(Z0 * dw[:, :, t], 1)
            tx=torch.hstack((self.times[t+1]*torch.ones((x.shape[0], 1)),x[:,:,t+1]))
            tx.requires_grad_()
            Y1=self.model(tx)
            Z1=torch.autograd.grad(Y1, tx, create_graph=True,grad_outputs=torch.ones_like(Y1),allow_unused=True)[0][:,1:]
            loss+=torch.sum(torch.square(Y1_tilde-Y1))
            Y0=Y1
            Z0=Z1
        
        loss += torch.sum(torch.square(Y1 - self.eqn.g_tf(x[:,:,-1])))
        x.requires_grad_()
        loss += torch.sum(torch.square(Z1 - self.Dg_tf(x[:,:,-1])))
        return loss
    
    def loss_fn2(self,inputs):
        dw,x=inputs
        inte=0.0
        for i in range(self.Ndis-1):
            tx=torch.hstack((self.times[i]*torch.ones((x.shape[0], 1)),x[:,:,i])).requires_grad_(True)
            Y=self.model(tx)
            Z=self.eqn.sig*torch.autograd.grad(Y, tx, create_graph=True,grad_outputs=torch.ones_like(Y),allow_unused=True)[0][:,1:]
            term1=-Z*dw[:,:,i]
            term2=self.eqn.f_tf(self.times[i],x[:,:,i],Y,Z).unsqueeze(dim=-1)*self.dt
            inte+=term1+term2

        return torch.mean(torch.square(inte-self.eqn.g_tf(x[:,:,-1]).unsqueeze(dim=-1)))
    

    