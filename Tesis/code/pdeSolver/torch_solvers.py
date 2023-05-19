import numpy as np
import torch
import torch.nn as nn
from torch_architectures import ResNetLikeDGM
from torch.optim.lr_scheduler import LambdaLR
import time

class DGM_solver():
    """
    Class for training a neural network for approximating a solution to a PDE
    using the DGM algorithm.
    """
    def __init__(self, eqn,solver_params):
        self.eqn=eqn
        self.initial_lr=solver_params["initial_lr"]
        self.model=ResNetLikeDGM(self.eqn.dim+1,1)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.initial_lr,weight_decay=0.00001)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,solver_params["lambda_lr"])
        self.Lweights=solver_params["initial_loss_weigths"]
        self.logging_interval=solver_params["logging_interval"]
        self.dtype=solver_params["dtype"]
        torch.set_default_dtype(self.dtype)
        self.currEps=0
        self.Nsamp=512
        self.interior_valid=torch.as_tensor(self.eqn.interior_point_sample(2048),dtype=self.dtype).requires_grad_()
        print(self.interior_valid.dtype)
        np_neu,np_n_neu=self.eqn.neumann_sample(2048,0)
        self.neumann_valid=(torch.as_tensor(np_neu,dtype=self.dtype).requires_grad_(),
                            torch.as_tensor(np_n_neu,dtype=self.dtype).requires_grad_())
        self.dirichlet_valid=torch.as_tensor(self.eqn.dirichlet_sample(2048,0),dtype=self.dtype).requires_grad_()
        self.terminal_valid=torch.as_tensor(self.eqn.terminal_sample(2048),dtype=self.dtype).requires_grad_()

    def set_lr(self,lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def save_model(self,file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self,file_name):
        self.model.load_state_dict(torch.load(file_name))

    def control(self,t,X):
        pos=torch.tensor(np.hstack((t,X)),requires_grad=True)
        u=self.model(pos)
        alpha=-np.sqrt(self.lambd)*torch.autograd.grad(u,pos)[0][1:]
        return alpha.detach().numpy()

    def loss(self,interior_sample,neumann_sample,dirichlet_sample,terminal_sample):

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

        return Li,Ln,Ld,LT
    
    def train(self,epochs):
        
        training_history = []
        print('Comencemos a calcular')
        start_time=time.time()

        # Begin iteration over samples
        for step in range(epochs):
            #print(step)
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
