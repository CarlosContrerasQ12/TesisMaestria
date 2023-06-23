import numpy as np
from juliacall import Main as jl
import time
from multiprocessing.pool import ThreadPool as Pool

def fa(p):
    dom=jl.EmptyRoom(0.4,0.6)
    resp=jl.simulate_one_path_Nagents(dom,0.01,0.001,0.0,1.0,[0.5,0.5,0.2,0.2,0.2,0.8],jl.Inf,3)
    return resp[0],resp[1],resp[2]

def main():
    jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/paths_EmptyRoom.jl")
    jl.seval("using .pathsEmptyRoom")
    pool = Pool()
    print(fa(2))
    print("Ya compilado")
    t0=time.time()
    with Pool(4) as pool:
        sap=pool.map(fa,range(100))
        pool.close()
        pool.join()
    print(sap[0])

if __name__ == "__main__":
    main()