#python version 2.7
import cPickle, gzip
import scipy as sp
import matplotlib.pyplot as plt
from collections import deque 
import scipy.ndimage as sn

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
        '''plots a digit. If nvals is true, then it uses a colormap to 
        distinguish negative and positive values.'''
        adigit=adigit.reshape(28,28)
        if nvals is True:
            plt.imshow(adigit,cmap="RdBu")
        else:
            plt.imshow(adigit,cmap="Greys")
        plt.show()
        
    def plotIdx(self, idx):
        '''plots a digit at the specified index in training set'''
        y = sp.reshape(self.train[0][idx], (28,28))
        plt.imshow(y, cmap = 'Greys')
        plt.show()
        
    def getSlice(self, dsname, aslice):
        '''used setattr() to assign values to attributes'''
        setattr(self, dsname, list(getattr(self, dsname)))
        getattr(self, dsname)[0] = getattr(self, dsname)[0][aslice]
        getattr(self, dsname)[1] = getattr(self, dsname)[1][aslice]
        setattr(self, dsname, tuple(getattr(self, dsname)))
            
class MNISTClassifierBase:
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
            #creating the list where len_vec is greater than desired value
            bool_list = list(map(lambda z: z>10, len_vec))
            #converting boolean list to array of 1 0
            bool_array = sp.array(bool_list, dtype = int)
            #calculating factor to be divided with
            norm_factor = sp.divide(bool_array, float(m)*len_vec)
            norm_factor[norm_factor == 0] = 1 #replacing 0's with 1
            temp_norm = sp.reshape(norm_factor, (len(norm_factor),1))*sp.ones(sp.shape(stoc_grad))
            stoc_grad = sp.divide(stoc_grad, temp_norm)
            '''alternatively we can use this
            for i in range(len(len_vec)): #this for loop is small as len_vec len is 10
                if len_vec[i] > 10:
                    stoc_grad[i,:] = stoc_grad[i,:]/(float(m)*float(len_vec[i]))'''
        return stoc_grad
        
    '''stack and grad_pooling are used in sq_loss and multinomial sub classes. They 
    are used to evaluate parts of func and grad values'''
    def stack(self, n, str_name):
        '''gives DSX784 matrix with data as index'''
        '''this function is called by the sub classes to vecctorize
        func evaluation and gradient evaluation (funcEval etc.) acc to data index'''
        temp = sp.vstack((getattr(self, str_name), self.x_temp[n, :]))
        setattr(self, str_name, temp)
        
    def grad_pooling(self, n): #consolidates gradient vector
        '''converts DSX784 to 10X784'''
        '''This will take the large gradient vector's column and add them according 
        to the index'''
        idx = [0,1,2,3,4,5,6,7,8,9]
        temp = sn.measurements.sum(self.grad_vec_l[:,n],self.dat_temp[1],idx)
        temp1 = sp.reshape(temp, (10,1))
        self.grad_vec = sp.hstack((self.grad_vec, temp1))
        
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
        result = [sp.argmin(sp.square(x - data1[0][i]).sum(axis = 1, dtype = float)) for i in range(len(data1[1]))]
        return sp.array(result) #1X50000 array
         
    def funcEval(self, x, data1):
        '''assuming x as a 10X784 matrix and every element in x is a float
        data is a tuple of pixel vectors and its label or what no the vector 
        represent'''
        self.fn = sp.ones(784) #next two steps are req for calling stack fn
        self.x_temp = x
        map(self.stack, data1[1], ['fn' for i in range(len(data1[1]))])
        self.fn = sp.delete(self.fn, (0), axis=0) #deleting the first row (of ones created above)
        fval_large = sp.square(self.fn - data1[0]) #DSX784, where DS is dataset size
        fval_large = fval_large.sum(axis = 1, keepdims = True) #DSX1
        fval = sn.measurements.sum(fval_large, data1[1], index = [0,1,2,3,4,5,6,7,8,9])
        fval = sp.reshape(fval, (10,1))
        return fval        
#==============================================================================
#         fval = sp.zeros(sp.shape(x))
#         for i in range(len(data1[1])):
#             fval[data1[1][i], :] += sp.square(x[data1[1][i], :] - data1[0][i])
#         '''it returns fval vector(10X1)'''
#         return fval.sum(axis = 1, keepdims = True)  #10X1 matrix  
#==============================================================================
    
    def gradEval(self, x, data1):
        #gradient is calculated in a vectorized way
        '''by calling stack fn we will create a giant matrix of variables that 
        mirrors the data. Then, we will apply the gradient operation i.e, 2*(w-data).
        Next, we will call the grad_pooling to add the relevant components; here 
        stacking is done horizontally (784 components are stacked horizontall to be exact)'''
        self.dat_temp = data1
        m = float(sp.shape(x)[1]) #no of columns
        self.gd = sp.ones(m)
        self.x_temp = x
        map(self.stack, data1[1], ['gd' for i in range(len(data1[1]))])
        self.gd = sp.delete(self.gd, (0), axis=0)#deleting the first row (of ones created above)
        self.grad_vec_l = 2*(self.gd - data1[0])
        self.grad_vec = sp.ones((10,1))
        iter_temp = sp.array([i for i in range(784)])
        map(self.grad_pooling, iter_temp)
        self.grad_vec = sp.delete(self.grad_vec, (0), axis=1)
        
        '''normalizing gradient vector if necessary
        len_vec is a array of length 10 or no of parameters'''
        len_vec = sp.sqrt(sp.diagonal(sp.dot(self.grad_vec, sp.transpose(self.grad_vec))))
        for i in range(len(len_vec)):
            if len_vec[i] > 1000:
                self.grad_vec[i,:] = m*self.grad_vec[i,:]/float(len_vec[i])
        return self.grad_vec #10X784 matrix
        
#==============================================================================
#Alternatively it can be written as         
#        for i in range(len(data1[1])):
#             grad_vec[data1[1][i], :] += 2*(x[data1[1][i], :] - data1[0][i])    
#==============================================================================
        
        
class MNISTMultiNom(MNISTClassifierBase):
    
    def classify(self, x, data1):
        '''for a point at every learned vector we will calculate the probability
        whichever is max(argmax) that index is assigned to the data point'''
        result = [sp.argmax(sp.exp((x*data1[0][i]).sum(axis =1, keepdims = True))) for i in range(len(data1[1]))]
        return sp.array(result)  #1X50000 array
        
    def secondpart(self, n):
        '''this fn calc the "sum" of exponentials in the multinomial fn'''
        temp = self.x_temp[n,:]*self.dat_temp[0]
        if self.fn2 == None:
            self.fn2 = sp.exp(temp.sum(axis=1, keepdims=True))
        else:
            self.fn2 += sp.exp(temp.sum(axis=1, keepdims=True))
        
    def funcEval(self, x, data1):
        '''assuming x as a 10X784 matrix and every element in x is a float
        data is a tuple of pixel vectors and its label or what no the vector 
        represent'''
        '''second or logarithm part of function is calc by taking log of secondpart fn
        return value. First part is calc with in this fn itself.'''
        #every thing is vectorized to calc fn value.
        self.x_temp = x
        self.dat_temp = data1
        self.fn2 = None
        map(self.secondpart, [i for i in range(10)])
        fn2_log = sp.log(self.fn2)
        
        #fn_stacked creates first parts w's
        self.fn_stacked = sp.ones(784) #next two steps are req for calling stack fn
        map(self.stack, data1[1], ['fn_stacked' for i in range(len(data1[1]))])
        self.fn_stacked = sp.delete(self.fn_stacked, (0), axis=0) #deleting the first row (of ones created above)
        fn1_temp = self.fn_stacked*self.dat_temp[0]
        fn1 = fn1_temp.sum(axis=1, keepdims=True)
        
        self.fval_long = -1*fn1 + fn2_log
        
        idx = [0,1,2,3,4,5,6,7,8,9]
        fval = sn.measurements.sum(self.fval_long,self.dat_temp[1],idx)
        fval = sp.reshape(fval, (10,1))
        
        return fval #10X1 matrix
        
#==============================================================================
#         fval = sp.zeros((int(sp.shape(x)[0]), 1))
#         for i in range(len(data1[1])):
#             '''wa is a scalar of wa, w(row) size is 1X784 and aT size is 784X1'''
#             wa = sp.sum(x[data1[1][i], :]*(data1[0][i]))
#             '''swa is a vector of 10X1 w is matrix of 10X784 and aT size is 784X1'''
#             swa = (x*(data1[0][i])).sum(axis = 1, keepdims = True)
#             fval[data1[1][i]] += sp.log(sp.sum(sp.exp(swa))) - wa
#         return fval  #10X1 matrix
#==============================================================================

    def gradEval(self, x, data1):
        '''gradient form is negative for the form shown in the image'''
        self.x_temp = x
        self.dat_temp = data1
        self.fn2 = None
        map(self.secondpart, [i for i in range(10)])
        gd2b = self.fn2 #DSX1
        
        #fn_stacked creates first parts w's
        self.gd_stacked = sp.ones(784) #next two steps are req for calling stack fn
        map(self.stack, data1[1], ['gd_stacked' for i in range(len(data1[1]))])
        self.gd_stacked = sp.delete(self.gd_stacked, (0), axis=0) #deleting the first row (of ones created above)
        
        gd2a1 = self.gd_stacked*self.dat_temp[0]
        gd2a = sp.exp(gd2a1.sum(axis=1, keepdims=True)) #DSX1
        
        temp = sp.divide(gd2a, gd2b) * self.dat_temp[0]
        
        self.grad_vec_l = -1*self.dat_temp[0] + temp
        
        self.grad_vec = sp.ones((10,1))
        iter_temp = sp.array([i for i in range(784)])
        map(self.grad_pooling, iter_temp)
        self.grad_vec = sp.delete(self.grad_vec, (0), axis=1)
        return self.grad_vec
        
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

#==============================================================================
# if __name__ == '__main__':
#     import timeit
#     from matplotlib import pylab
#     start = timeit.default_timer()
#     func = MNISTMultiNom()
#     #func1 = MNISTSqloss()
#     x0 = sp.random.randn(10,784)
#     solution = SGD(func, x0, 100, 0.9)
#     solution.getSoln()
#     w = solution.history_x.pop()
#==============================================================================
    
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

    
    
    
    
    
            
        
            
            
        
        
            
            
        
    
        
    
