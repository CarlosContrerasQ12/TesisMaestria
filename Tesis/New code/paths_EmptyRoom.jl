struct EmptyRoom
    pInf::Float64
    pSup::Float64
    pAnc::Float64
    EmptyRoom(pInf,pSup)=new(pInf,pSup,pSup-pInf)
end

function exited_domain(dom::EmptyRoom,X0,X1)
    if X1[1]<0.0 || X1[1]>1.0 || X1[2]>1.0 || X1[2]<0.0
        v=X1-X0
        tleft=-X0[1]/(v[1])
        tup=(1-X0[2])/(v[2])
        trig=(1-X0[1])/(v[1])
        tdow=-X0[2]/(v[2])
        ts=[tleft,tup,trig,tdow]
        ts=@. ifelse(ts > 1.0, 10.0, ts)
        ts=@. ifelse(ts < 0.0, 10.0, ts)
        i=argmin(ts)
        if i==1
            Xi=[0.0,X0[2]+ts[i]*(v[2])]
            return Xi,"Neu",ts[i]     
        elseif i==2
            Xi=[X0[1]+ts[i]*(v[1]),1.0]
            return Xi,"Neu",ts[i]
        elseif i==3
            Xi=[1.0,X0[2]+ts[i]*(v[2])]
            if Xi[2]>dom.pInf && Xi[2]<dom.pSup
                return Xi,"Dir",ts[i]
            else
                return Xi,"Neu",ts[i]
            end
        elseif i==4
            Xi=[X0[0]+ts[i]*(v[0]),0.0]
            return Xi,"Neu",ts[i]
        end
    end
    return X1,"Non",1.0
end

function one_agent_brownian(dom::EmptyRoom,sig,dt,path_length,X0,dirichlet_cut=false,neumann_cut=false)
    sqdt=sqrt(dt)
    X=zeros((path_length,2))
    X[1,:]=X0
    xis=randn((path_length-1,2))
    dtf=1.0
    for i in 2:path_length
        Xn=X[i-1,:]+sqdt*sig*xis[i-1,:]
        Xnew,exit,dtf=exited_domain(dom,X[i-1,:],Xn)
        if exit=="Neu"
            if !neumann_cut
                Xn=Xnew+sqrt(dt*(1-dtf))*sig*randn(2)
                Xs,e,dtf2=exited_domain(dom,Xnew,Xn)
                while e!="Non"
                    Xn=Xnew+sqrt(dt*(1-dtf))*sig*randn(2)
                    Xs,e,dtf2=exited_domain(dom,Xnew,Xn)
                end
                X[i,:]=Xs
                xis[i-1,:]=(X[i,:]-X[i-1,:])/(sqdt*sig)
            else
                X[i,:]=Xnew
                return X[begin:i],xis[begin:i-1],dtf
            end
        elseif exit=="Dir"
            if !dirichlet_cut
                X[i,:]=Xnew
                X[i+1:end,:]=[1.0,dom.pInf+0.5*dom.pAnc]
                xis[i:end,:].=0.0
                return X,xis,dtf
            else
                X[i,:]=Xnew
                return X[begin:i],xis[:i-1],dtf
            end
        elseif exit=="Non"
            X[i,:]=Xnew
        end
    end
    return X,xis,dtf
end

dom=EmptyRoom(0.4,0.6)
X,xis,dtf=one_agent_brownian(dom,0.01,0.01,10,[0.5,0.5],true,false);

X