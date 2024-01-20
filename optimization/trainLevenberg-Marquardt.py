# Levenberg-Marquardt Optimization

import numpy as np

# objective Extended Rosenbrock function
def objfcn(x):
    # Minima -> f=0 at (1,.....,1)
    n = len(x) # n even
    fvec = np.zeros((n,1))
    idx1 = np.array(range(0,n,2)) # odd index
    idx2 = np.array(range(1,n,2)) # even index
    fvec[idx1]=10.0*(x[idx2]-(x[idx1])**2.0)
    fvec[idx2]=1.0-x[idx1]
    f = fvec.T @ fvec
    return f[0,0]

# Extended Rosenbrock gradient function
def objfcngrad(x):
    n = len(x) # n even
    Jf = np.zeros((n,n))
    fvec = np.zeros((n,1))
    idx1 = np.array(range(0,n,2)) # odd index
    idx2 = np.array(range(1,n,2)) # even index
    fvec[idx1]=10.0*(x[idx2]-(x[idx1])**2.0)
    fvec[idx2]=1.0-x[idx1]

    for i in range(n//2):
        Jf[2*i,2*i]     = -20.0*x[2*i]
        Jf[2*i,2*i+1]   = 10.0
        Jf[2*i+1,2*i]   = -1.0

    gX = 2.0*Jf.T @ fvec
    return gX

def objfcnjac(x):
    # Extended Rosenbrock Jacobian Function
    n = len(x) # n even
    Jf = np.zeros((n,n))
    fvec = np.zeros((n,1))
    idx1 = np.array(range(0,n,2)) # odd index
    idx2 = np.array(range(1,n,2)) # even index
    fvec[idx1]=10.0*(x[idx2]-(x[idx1])**2.0)
    fvec[idx2]=1.0-x[idx1]

    for i in range(n//2):
        Jf[2*i,2*i]     = -20.0*x[2*i]
        Jf[2*i,2*i+1]   = 10.0
        Jf[2*i+1,2*i]   = -1.0

    gX = 2.0*Jf.T @ fvec
    normgX = np.linalg.norm(gX)
    return fvec, Jf, normgX

# Levenberg-Marquardt Optimization
def trainlm(X,maxEpochs,goal,mu,mu_dec,mu_inc,mu_max,mingrad,show):
    # trainlm is a optimization function that updates variables
    # values according to Levenberg-Marquardt
    this = "trainlm"
    stop = ""
    epochs = []
    perfs  = []
	# generate an initial point
    n = len(X)
    I = np.eye(n)
    perf = objfcn(X)
    print("\n")
    # Train
    for epoch in range(maxEpochs+1):
        # calculate function equations (F), jacobian matrix (J) and
        #  gradient value (normgX)
        F, J, normgX = objfcnjac(X)

        # Stopping criteria
        if perf <= goal:
            stop = "Performance goal met."
        elif epoch == maxEpochs:
            stop = "Maximum epoch reached, performance goal was not met."
        elif normgX < mingrad:
            stop = "Minimum gradient reached, performance goal was not met."
        elif mu > mu_max:
            stop = "Maximum MU reached, performance goal was not met."
        
        # Progress
        if (np.fmod(epoch,show) == 0 or len(stop) != 0):
            print(this,end = ": ")
            if np.isfinite(maxEpochs):
                print("Epoch ",epoch, "/", maxEpochs,end = " ")
            if np.isfinite(goal):
                print(", Performance %8.3e" % perf, "/", goal, end = " ")
            if np.isfinite(mingrad):
                print(", Gradient %8.3e" % normgX, "/", mingrad)

            epochs = np.append(epochs,epoch)
            perfs = np.append(perfs,perf)

            if len(stop) != 0:
                print("\n",this,":",stop,"\n")
                break
       
        # Levenberg Marquardt
        while mu <= mu_max:
            JTF = -J.T @ F
            JTJ = J.T @ J
            H = JTJ + mu*I
            # Equation solve H*dX = JF
            dX = np.linalg.solve(H,JTF) 
            X2 = X + dX
            perf2 = objfcn(X2)
            if perf2 < perf:
                X = X2
                perf = perf2
                mu = mu * mu_dec
                break
            mu = mu * mu_inc

    return X, perfs, epochs

def main():
    X = np.repeat([10],1000).reshape(1000,1)
    # Performance goal met
    goal = 1e-8
    # define the total iterations
    max_epochs = 1000
    # rate learning
    mu = 0.1
    # rate learning decrement
    mu_dec = 0.1
    # rate learning increment
    mu_inc = 10
    # rate learning maximum
    mu_max = 1e10
    # minimum gradient
    min_grad = 1e-11
    # show
    show = 10
    # perform the Levenberg Marquardt
    X, perfs, epochs = trainlm(X,max_epochs,goal,mu,mu_dec,mu_inc,mu_max,min_grad,show)
    print(X)
