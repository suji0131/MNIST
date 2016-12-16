import cPickle, gzip
import scipy as sp
import matplotlib.pyplot as plt
from collections import deque 

class MNISTDataSet:
    def __init__(self, dsrange = None):
        '''Reads in the data file. dsrange is a slice object. If
        dsrange is not None returns we drop every thing from the 
        training set other than that slice.'''
        f = gzip.open('mnist.pkl.gz', 'rb')
        self.train, self.validation, self.test = cPickle.load(f)
        f.close()
        if dsrange != None:
            self.getSlice('train', dsrange)
        
    def plotDigit(self, adigit, nvals=False):
        #plots a digit
        adigit=adigit.reshape(28,28)
        if nvals is True:
            plt.imshow(adigit,cmap="RdBu")
        else:
            plt.imshow(adigit,cmap="Greys")
        plt.show()
        
    def plotIdx(self, idx):
        #plots a digit at the specified index in training set
        y = sp.reshape(self.train[0][idx], (28,28))
        plt.imshow(y, cmap = 'Greys')
        plt.show()
        
    def getSlice(self, dsname, aslice):
        getattr(self, dsname) = list(getattr(self, dsname))
        getattr(self, dsname)[0] = getattr(self, dsname)[0][aslice]
        getattr(self, dsname)[1] = getattr(self, dsname)[1][aslice]
        getattr(self, dsname) = tuple(getattr(self, dsname)
            
class MNISTClassifierBase():
    def __init__(self, ds = None):
        '''ds should always be a MNISTDataSet object'''
        self.data = MNISTDataSet()
        if ds != None:
            self.data = ds
    
    def errorInd(self, x):
        '''calculates the no of mis-classification in test set'''
        a = self.classify(x, self.data.test)
        b = self.data.test[1]
        return sp.count_nonzero(b.astype(int)-a.astype(int))
    
    def feval(self, x, average = True):
        '''I'm considering opt problem regarding parameters of each digit 
        independently i.e.., I have ten opt problems to solve. 
        loss_fn is an 10X1 matrix or vector'''
        data = self.data.train
        loss_fn = self.funcEval(x, data)
        if average == True:
            n_vals = sp.bincount(self.data.train[1]).astype(float) #counts no of 1's 2's ..... in the dataset
            loss_fn = sp.divide(loss_fn, sp.reshape(n_vals, sp.shape(loss_fn)))
            return loss_fn
        else:
            return loss_fn
            
    def sfeval(self, x, ndata = 100, average = True):
        '''slice the data, slice size is given by ndata.'''
        low_no = sp.random.randint(0, high = 50000 - ndata)
        high_no = low_no + ndata
        sli = slice(low_no, high_no)
        data2 = []
        data2.append(self.data.train[0][sli])
        data2.append(self.data.train[1][sli])
        loss_fn = self.funcEval(x, tuple(data2))
        if average == True:
            '''assuming that in this 100 numbers every digit happens atleast once'''
            n_vals = sp.bincount(data2[1]).astype(float) 
            loss_fn = sp.divide(loss_fn, sp.reshape(n_vals, sp.shape(loss_fn)))
            return loss_fn
        else:
            return loss_fn
        
    def grad(self, x):
        return self.gradEval(x, self.data.train)
        
    def sgrad(self, x, ndata = 100, bound = True, average = True):
        low_no = sp.random.randint(0, high = 50000 - ndata)
        high_no = low_no + ndata
        sli = slice(low_no, high_no)
        data2 = []
        data2.append(self.data.train[0][sli])
        data2.append(self.data.train[1][sli])
        gradient = self.gradEval(x, tuple(data2))
        m = int(sp.shape(gradient)[1])
        '''u is the random direction matrix'''
        u = sp.random.randint(0, high = m, size = sp.shape(gradient))
        '''taking element wise product of u and gradient and then row wise sum to 
        get dot product of the matrices. dotprod is 10X1 matrix'''     
        dotprod = (u*gradient).sum(axis=1, keepdims = True)
        stoc_grad = dotprod*u
        if bound == True:
            len_vec = sp.sqrt(sp.diagonal(sp.dot(stoc_grad, sp.transpose(stoc_grad))))
            for i in range(len(len_vec)): #this for loop is small as len_vec len is 10
                if len_vec[i] > 10:
                    stoc_grad[i,:] = stoc_grad[i,:]/(float(m)*float(len_vec[i]))
        return stoc_grad
        
    def classify(self, x, data1):
        pass
        
    def funcEval(self, x, data1):
        pass
            
    def gradEval(self, x, data1):
        pass
        
    
class MNISTSqloss(MNISTClassifierBase):

    def classify(self, x, data1):
        '''for every point its distance from the parameters is calculated and 
        argmin of the array is returned'''
        result = []
        for i in range(len(data1[1])):
            temp = sp.square(x - data1[0][i]).sum(axis = 1, dtype = float)
            result.append(sp.argmin(temp))
        return sp.array(result) #1X50000 array
                
    def funcEval(self, x, data1):
        '''assuming x as a 10X784 matrix and every element in x is a float
        data is a tuple of pixel vectors and its label or what no the vector 
        represent'''
        fval = sp.zeros(sp.shape(x))
        for i in range(len(data1[1])):
            fval[data1[1][i], :] += sp.square(x[data1[1][i], :] - data1[0][i])
        '''it returns fval vector(10X1)'''
        return fval.sum(axis = 1, keepdims = True)  #10X1 matrix  
     
    def gradEval(self, x, data1):
        m = float(sp.shape(x)[1]) #no of columns
        grad_vec = sp.zeros(sp.shape(x)) #grad_vec size will be 10X784
        for i in range(len(data1[1])):
            grad_vec[data1[1][i], :] += 2*(x[data1[1][i], :] - data1[0][i])    
        '''normalizing gradient vector if necessary
        len_vec is a array of length 10 or no of parameters'''
        len_vec = sp.sqrt(sp.diagonal(sp.dot(grad_vec, sp.transpose(grad_vec))))
        for i in range(len(len_vec)):
            if len_vec[i] > 1000:
                grad_vec[i,:] = m*grad_vec[i,:]/float(len_vec[i])
        return grad_vec #10X784 matrix
        
        
class MNISTMultiNom(MNISTClassifierBase):
    
    def classify(self, x, data1):
        '''for a point at every learned vector we will calculate the probability
        whichever is max(argmax) that index is assigned to the data point'''
        result = []
        for i in range(len(data1[1])):
            temp = sp.exp((x*data1[0][i]).sum(axis =1, keepdims = True))
            result.append(sp.argmax(temp))
        return sp.array(result)  #1X50000 array
        
    def funcEval(self, x, data1):  
        fval = sp.zeros((int(sp.shape(x)[0]), 1))
        for i in range(len(data1[1])):
            '''wa is a scalar of wa, w(row) size is 1X784 and aT size is 784X1'''
            wa = sp.sum(x[data1[1][i], :]*(data1[0][i]))
            '''swa is a vector of 10X1 w is matrix of 10X784 and aT size is 784X1'''
            swa = (x*(data1[0][i])).sum(axis = 1, keepdims = True)
            fval[data1[1][i]] += sp.log(sp.sum(sp.exp(swa))) - wa
        return fval  #10X1 matrix
        
    def gradEval_behind( self, anarray):
        multiplied_array=sp.exp(sp.sum(self.y * anarray, axis=1))
        numerator=multiplied_array[self.i]
        denominator = sp.sum(multiplied_array)
        return numerator/denominator*anarray

    def gradEval(self, x, data):
        self.y=x
        grad = []
        for i in range(0,10):
            self.i = i
            scalar1 = sp.sum(data[0][data[1] == i],axis=0)
            scalar2 = sp.sum(sp.apply_along_axis(self.gradEval_behind,1,data[0]),axis=0)
            grad.append(scalar2 - scalar1)
        grad_vec =  sp.array(grad).reshape((10,784)) 
        return grad_vec #10X784 matrix
        
class SGD:
    def __init__(self, afunc, x0, inistepsize, gamm, proj=None, histsize=-1, smallhist=False, ndata = 100, keepobj = True):
        self.afunc = afunc
        self.x0 = x0
        self.x_reset = x0
        self.step0 = inistepsize
        self.gamma = gamm
        self.proj= proj 
        self.histsize= histsize 
        self.smallhist= smallhist 
        self.ndata = ndata
        self.keepobj = keepobj
        self.f_new = 100000000 + sp.zeros_like(self.x0[:,0], dtype = float)
        self.n = 0
        if self.histsize == -1:
            self.history_x = deque()
            self.history_f = deque()
            self.history_x.append(self.x0)
            self.history_f.append(self.afunc.sfeval(x0))
        else:
            self.history_x = deque(maxlen = self.histsize)
            self.history_x.append(self.x0)
            self.history_f.append(self.afunc.sfeval(x0))
    
    def step_size(self):
        return self.step0/self.n**self.gamma
        
    def reset(self):
        self.n = 0
        self.x0 = self.x_reset
        self.history_x.clear()
        self.history_f.clear()
        self.f_new = 100000000 + sp.zeros_like(self.x0[:,0], dtype = float)
        
        
    def setStart(self, x0):
        self.x0 = x0
        return self.x0
        
    def dostep(self):
        
        self.new_x = self.x0 - self.step_size()*self.afunc.sgrad(self.x0)
        self.history_x.append(self.new_x)
        self.history_f.append(self.afunc.sfeval(self.new_x))
        self.x0 = self.setStart(self.new_x)
        
    def nsteps(self, an):
        nsteps = an
        for i in range(nsteps):
            self.n = self.n + 1
            self.dostep()
            
        
    def getAvgSoln(self, wsize = 10):
        avg_sum = sp.zeros_like(self.x0[:, 0], dtype = float)
        for i in range(wsize):
            avg_sum = avg_sum + self.history_f[len(self.history_f) -1 - i]
        return avg_sum/wsize
            
    def getSoln(self, wsize = 10, winterval = 1, abstol = 1e-6, reltol = 1e-6):
        nreq_steps = (winterval + 1)*wsize
        self.f_old = sp.zeros_like(self.x0[:,0], dtype = float)
        
        while self.n < 3000:
            self.f_old = self.f_new
            self.nsteps(nreq_steps)
            self.f_new = self.getAvgSoln()

        return self.history_f[len(self.history_f) -1]
        
if __name__ == '__main__':
    import timeit
    from matplotlib import pylab
    start = timeit.default_timer()
    func = MNISTMultiNom()
    #func1 = MNISTSqloss()
    x0 = sp.random.randn(10,784)
    solution = SGD(func, x0, 100, 0.9)
    solution.getSoln()
    w = solution.history_x.pop()
    
#algorithm is taking lot of minutes(approx 5 mins) to converge and classification 
#errors are in range of 80 to 90 for both classifiers with multinomial performing better
#==============================================================================
#     errors = func.errorInd(w)
#     time_taken = timeit.default_timer() - start
#     print "Train and test time  " + str(time_taken)  +"  (sec)"
#     print " Multi Nom Error  " + str (errors) 
#==============================================================================
    
# printing gradient for 3 for 4's gradient changed the loss fn and idx(use cmap = 'seismic') 
#==============================================================================
#     graddigit=func.gradEval(sp.zeros((10,784)),(func.data.train[0][0:51],func.data.train[1][0:51]))
# 
#     plt.imshow(graddigit[3].reshape(28,28),cmap=pylab.matplotlib.cm.bwr)
#     plt.colorbar()
#     plt.show()
#==============================================================================
    
#printing digit
#==============================================================================
# ds = MNISTDataSet()
# ds.plotIdx(0)
#==============================================================================

    
    
    
    
    
            
        
            
            
        
        
            
            
        
    
        
    
