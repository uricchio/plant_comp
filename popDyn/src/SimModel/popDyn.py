import sys
import os
import numpy as np
import random
import math
from collections import defaultdict
from scipy.stats import nbinom
from scipy.stats import binom
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.special import digamma
from scipy.optimize import minimize
from scipy.optimize import brentq

# NEXT: import lines from posteriors to account for uncertainty

class MetaPop:

    def __init__(self,numSpec=5):

        # order is SP,EG,BH,BD,AB
        self.sp = ''
        self.eg = ''
        self.bh = ''
        self.bd = ''
        self.ab = ''
        self.sc_pos = 0
        self.data = defaultdict(dict)
        self.headers = defaultdict(dict)
        self.numSpec = numSpec        
        #self.species = np.zeros(numSpec)
        self.species = np.array([10,10,0,0,0])
        self.speciesEig = np.array([10,10,0,0,0])
        self.compParams = np.zeros((numSpec,numSpec))
        self.negBinP = [0.001058495, 0.001028352, 0.017663834, 0.029322691, 0.020070808]
        self.negBinR = [0.1864779, 0.1201766, 1.2604776, 1.5082897, 2.4441944]
        self.perSurv = [0.905,0.919,0.,0.,0.]
        self.seedToAdult = 1 # transition is already included elsewhere
        self.numSeeds = [0,0,0,0,0]
        self.annuals = np.array([0,0,1.,1.,1.])
        self.labels = {}
        self.BFODmat = np.subtract(np.ones(5), [0.04566667, 0.01533333,0.02066667, 0.02333333,0.51466667])
        self.lam = [175.98616, 116.74312, 70.09875, 49.92934, 119.33438] # this is depricated and not used in any working functions
        self.germMat =  [0.2786667, 0.5426667, 0.5386667, 0.6226667, 0.3266667] # reverse of germ.pars$germ in Erin's R script 2016 Bayesian Analysis
        self.germMatUninf = [0.08720717, 0.27684292, 0.23950701, 0.23431707, 0.05046026][::-1] # germ.pars$germ*germ.pars$est.path in Erin's R script 2016 Bayesian Analysis
        self.germMatInf =  [0.08816386, 0.24118760, 0.23044574, 0.22765694, 0.05188997][::-1] #germ.pars$germ*germ.pars$est.path
        self.pInf = [0.2236517, 0.4445375, 0.1768596, 0.4297690, 0.5113082] # exp.path[,"P"]/(exp.path[,"P"]+exp.path[,"A"])
        self.perTrans = [1.,1.,1.,1.]
        self.infMat = np.zeros(5)
        self.meanDam = [0.0202,0.00860,0.01102,0.00523,0.00817]   
        self.seedlings = np.zeros(5)     

    def sampleData(self,spec,row=None,remove_spec = None):
        data = [[],[],[],[],[]]
        i = 0
        for tag in [self.sp,self.eg,self.bh,self.bd,self.ab]:
            if tag == self.sc or tag == self.dam:
                continue
            data[i] = self.data[spec][tag][:]
            i += 1
        data = np.array(data)
        data = data.T
        if not row:
            row = np.random.choice(len(data))
        for i in range(len(data[row])):
            data[row][i] = int(round(data[row][i]))
        self.species = data[row][:]
        for i in range(0,len(self.species)):
            if self.species[i] < 0.001:
                self.species[i] += 1.
        if remove_spec:
            self.species[remove_spec] = 0
        #self.numSeeds = np.multiply(self.species,np.divide(np.multiply(self.negBinR,np.subtract(np.ones(self.numSpec),self.negBinP)),self.negBinP))[:]
        #compMat  = np.exp(-1.*np.dot(self.compParams,self.species))
        compMat  = np.divide(1.,np.add(1.,np.dot(self.compParams,self.species)))
        for i in range(self.numSpec):
           self.numSeeds[i] = np.sum(np.multiply(self.get_r(compMat,i)*(1-self.negBinP[i])/(self.negBinP[i]),self.species[i]))


    def get_surv(self,file_eg,file_sp,egnum,spnum):
        egh = open(file_eg,'r')
        i = 0
        for line in egh:
            if i < egnum:
                i += 1
                continue
            line = line.strip().split()
            self.perSurv[1] = float(line[1])
            break
        sph = open(file_sp,'r')
        i = 0
        for line in sph:
            if i < spnum:
                i += 1
                continue
            line = line.strip().split()
            self.perSurv[0] = float(line[1])
            break
   
    def get_headers(self,f):
        fh = open(f,'r') 
        labels = []
        for line in fh:
            line = line.strip().split(',')
            #if len(line) == 1:
            #    line = line[0].strip().split()
            #print line
            i = 0
            for thing in line:
                if "\"" in thing:
                    thing = thing[1:-1]
                if thing == self.sc:
                    self.sc_pos = i
                if f not in self.labels:
                    self.labels[f] = []
                self.labels[f].append(thing)
                if f not in self.headers:
                    self.headers[f][thing] = []
                self.headers[f][thing] = []
                i += 1
            break
        fh.close()
        return
    
    def get_data(self,f,spec):
        fh = open(f,'r')
        data = {}
        for line in fh:
            break
        for line in fh:
            locdata = line.strip().split(',')
            #if len(locdata) == 1:
            #    locdata = line.strip().split()
            for k in range(len(locdata)):
                if "\"" in locdata[k]:
                    locdata[k] = locdata[k][1:-1] 
            if locdata[self.sc_pos] == "NA" or locdata[self.sc_pos] == "nan":
                continue
            #if float(locdata[self.sc_pos]) == 0.: 
            #    continue
            i = 0
            notspec = 0
            for thing in self.labels[f]:
                if thing == "Species" or thing == "spcode":
                    if locdata[i] != spec:
                        notspec = 1
                    break
                i += 1
            if notspec > 0:
                continue
            
            i = 0
            for thing in self.labels[f]:
                if thing != self.sc and thing != self.ab and thing != self.eg and thing != self.bh and thing != self.bd and thing != self.sp and thing != self.dam:
                    i += 1
                    continue
                if thing not in data:
                    data[thing] = []
                if locdata[i] == "NA":
                    locdata[i] = "NaN"
                try:
                    data[thing].append(float(locdata[i]))
                except:
                    data[thing].append(locdata[i])
                i += 1
        if spec not in self.data:
            self.data[spec] = data
        else:
            for tag in data:
                if tag in self.data[spec]:
                    self.data[spec][tag] = self.data[spec][tag][:]+data[tag][:]
                else:
                    self.data[spec][tag] = data[tag][:]
        fh.close()
    
    def map_labels(self,labels):
        self.sp  = labels[0]
        self.eg  = labels[1]
        self.bh  = labels[2]
        self.bd  = labels[3]
        self.ab  = labels[4]
        self.dam  = labels[5]
        self.sc = labels[6]

    def readPosteriorComp(self,postfile,i,row=None):
        ph = open(postfile,'r')
        postData = []
        for line in ph:
            data = [np.float(x) for x in line.strip().split()]
            postData.append(data)
        if not row: 
            row = np.random.randint(len(postData))
        self.negBinR[i] = postData[row][0]
        self.negBinP[i] = postData[row][1]
        #self.infMat[i] = math.exp(-postData[row][13]*self.meanDam[i]) #1./(1.+postData[row][13]*self.meanDam[i])
        self.infMat[i] = 1./(1.+postData[row][13]*self.meanDam[i])
        self.compParams[i] = postData[row][8:13][:]

    def readPerTrans(self,perfile,row=None):
        # perfile consists of 4 colums: SeedOnSeed, AnnualOnSeedling, PerOnSeedling, Intercept
        # comp function is Intercept*1/(1+ SeedOnSeed*Seed + AnnualOnSeedling*Ann + PerOnSeedling*Per)
        ph = open(perfile,'r')
        perData = []
        for line in ph:
            data = [np.float(x) for x in line.strip().split()]
            perData.append(data)
        if not row: 
            row = np.random.randint(len(perData))
        self.perTrans = perData[row][:]

    def addSeeds(self,nSeeds,i):
        self.numSeeds[i] += nSeeds
    
    def addSpec(self,nSpec,i):
        self.species[i] += nSpec

    def readPosteriorsAll(self,postfiles,row=None):
        # requires files to be in correct order!!
        i = 0
        for postfile in postfiles:
            self.readPosteriorComp(postfile,i,row)
            i += 1

    def readPosteriorsAnn(self,postfiles,row=None):
        # requires files to be in correct order!!
        i = 2
        for postfile in postfiles[2:]:
            self.readPosteriorComp(postfile,i,row)
            i += 1

    #def initCompParamsDefault(self):
    #    # taken from the mean of the posterior estimates for each species
        # majorly depricated since the estimates have changed
    #    self.compParams[0] = [2.146959e-03 ,1.516291e-02, 9.290507e-05, 1.242410e-03,7.553041e-04]
    #    self.compParams[1] = [6.661486e-03, 2.847138e-03, 8.497073e-05, 1.988545e-04,3.540785e-04]
    #    self.compParams[2] = [0.0910327289, 0.0206165758, 0.0002010239, 0.0063383113,0.0051127981]
    #    self.compParams[3] = [0.067119857, 0.052216943, 0.000693937, 0.000859324,0.005673260]
    #    self.compParams[4] = [0.316043761, 0.081228982, 0.001181376, 0.009999236,0.020203021]

    #def initInfParams(self):
        # taken from the mean of the posterior estimates for each species, multiplied by mean damage per species
    #    self.infMat = np.divide(np.ones(5),np.add(np.ones(5),np.array([0.004004548, 0.006937243, 0.2010776, 0.03431616, 0.0831712])))[:]
 
    def initCompParams(self,compParams=np.array([0.,0.,0.,0.,0.]),i=0):
        self.compParams[i] = compParams[:]

    def rescaleParams(self,C,i):
        # the point of this function is to rescale such that the monoculture 
        # density is consistent with the observed monoculture density
        # but may not be consistent with observed seed output data
        al_scaled = (-1+self.germMatUninf[i]*self.lam[i])/(C*self.germMatUninf[i])
        scale = al_scaled/self.compParams[i][i]
        self.compParams[i] = np.multiply(scale,(self.compParams[i]))[:]
        #self.negBinR[i] *= scale
        #print self.compParams[i], self.negBinR[i]
        #return

    def rescaleAnnParams(self, Cs):
        i  = 2
        for C in Cs[2:]:
            self.rescaleParams(C,i)
            i += 1
    
    def rescaleAllParams(self, Cs):
        i =0
        for C in Cs:
            self.rescaleParams(C,i)
            i += 1

    def get_r(self,compMat,i):
        ret = np.multiply(compMat[i],self.negBinR[i])
        if ret  == 0.:
            ret = 10**-20       
        return ret

    def nextGenStoch(self,inf=True,BFOD=True):
        # this function is currently depricated
        # some of these seeds die from BFOD
        if BFOD:
            self.numSeeds = binom.rvs(n=self.numSeeds,p=self.BFODmat)[:]
        # some of remaining seeds get pathogen infected
        germ = []
        if inf:
            infSeeds = binom.rvs(n=self.numSeeds,p=self.pInf)[:]
            unInfSeeds = np.subtract(self.numSeeds,infSeeds)[:]
            # some of seeds germinate
            germ = binom.rvs(p=self.germMatUninf,n=unInfSeeds)[:]
            germ = np.add(germ,binom.rvs(p=self.germMatInf,n=infSeeds)[:])[:]
        else:
            germ = binom.rvs(p=self.germMatUninf,n=self.numSeeds)[:]
        # subtract out germinated seeds from total number of seeds to keep in seed bank
        self.numSeeds = np.subtract(self.numSeeds,germ)[:]      
        # incorporate perennials that survived
        self.species = np.add(binom.rvs(p=self.perSurv,n=self.species.astype(int)).astype(int),binom.rvs(p=self.annuals,n=germ.astype(int).astype(int))).astype(int)[:]
        # competition
        compMat  = np.exp(-1.*np.dot(self.compParams,self.species))
        num_seeds_this_gen = np.zeros(self.numSpec)
        for i in range(self.numSpec):
            num_seeds_this_gen[i] += np.sum(nbinom.rvs(n=self.get_r(compMat,i),p=self.negBinP[i],size=self.species[i]))
        # infection
        if inf:
            num_seeds_this_gen = binom.rvs(p=self.infMat,n=num_seeds_this_gen.astype(int))[:]
        # add in new seeds to model
        self.numSeeds = np.add(self.numSeeds,num_seeds_this_gen.astype(int))[:]
        # add seedlings from perrenials to seedlings, and ones that survived from last year to adults
        self.seedlings = np.array([germ[0],germ[1],0,0,0])[:]
        self.species  = np.add(self.species,binom.rvs(p=self.perTrans[3]/(1.+(self.seedlings[0]+self.seedlings[1])*self.perTrans[0]+(self.species[0]+self.species[1])*self.perTrans[2]
                               +(self.species[2]+self.species[3]+self.species[4])*self.perTrans[1]),n=self.seedlings.astype(int)).astype(int))[:]
        #i = 0
        #for spec in num_seeds_this_gen:
        #    print spec,self.species[i], " ",
        #    i += 1
        #print
        return (self.species,self.numSeeds)
    
    def nextGen(self,inf=True,BFOD=True):
         
        newSeeds = self.numSeeds[:]

        # some of these seeds die from BFOD
        if BFOD:
            newSeeds = np.multiply(newSeeds,self.BFODmat)[:]
        
        survSeeds = newSeeds[:]

        # get total seeds that germinated (do not necessarily establish)
        germNotEst = np.multiply(newSeeds,self.germMat)[:]
        
        # subtract out germinated seeds from total number of seeds to keep in seed bank
        newSeeds  = np.subtract(newSeeds,germNotEst)[:]      

        # some of germinated seeds get pathogen infected, and some establish with or without infection
        est = []
        if inf:
            # proportion of infected seeds
            infSeeds = np.multiply(survSeeds,self.pInf)[:]
            unInfSeeds = np.subtract(survSeeds,infSeeds)[:]
            # some of seeds establish
            est = np.multiply(self.germMatUninf,unInfSeeds)[:]
            est = np.add(est,np.multiply(self.germMatInf,infSeeds)[:])[:]
        else:
            est = np.multiply(self.germMatUninf,survSeeds)[:]
       
        annuals = est[:]

        #split out perennial seedlings for clarity
        self.seedlings = [annuals[0],annuals[1],0,0,0]
        annuals = [0,0,annuals[2],annuals[3],annuals[4]]
        
        # incorporate perennials that survived
        perennials = np.multiply(self.perSurv,self.species)[:]
        
        self.species = np.add(perennials,annuals)[:]
        
        # competition
        #compMat  = np.exp(-1.*np.dot(self.compParams,self.species))
        compMat  = np.divide(1.,np.add(1.,np.dot(self.compParams,self.species)))
        
        num_seeds_this_gen = np.zeros(self.numSpec)
        for i in range(self.numSpec):
            num_seeds_this_gen[i] += np.sum(np.multiply(self.get_r(compMat,i)*(1-self.negBinP[i])/(self.negBinP[i]),self.species[i]))
        #print num_seeds_this_gen    
        # infection
        if inf is True:
            num_seeds_this_gen = np.multiply(self.infMat,num_seeds_this_gen)[:]
        
        self.numSeeds = np.add(num_seeds_this_gen,newSeeds)[:]
        #self.numSeeds = num_seeds_this_gen[:]

        # add seedlings from perrenials to seedlings, and ones that survived from last year to adults
        surv = self.perTrans[3]/(1.+(self.seedlings[0]+self.seedlings[1])*self.perTrans[0]+
                  (self.species[0]+self.species[1])*self.perTrans[2]+(annuals[2]+annuals[3]+annuals[4])*self.perTrans[1])
        
        self.speciesEig = self.species[:]
        self.species = np.add(self.species,np.multiply(surv*self.seedToAdult,self.seedlings))[:]

        return (self.species,num_seeds_this_gen)
        
    def compEigen(self,i,inf=True,BFOD=True):
        # perennial tranisition matrix, described at
        # http://www.esapubs.org/archive/ecol/E094/254/appendix-A.php
        #compMat  = np.exp(-1.*np.dot(self.compParams,self.speciesEig))
        compMat  = np.divide(1.,np.add(1.,np.dot(self.compParams,self.speciesEig)))
        germ = []
        newseeds = np.zeros(5)
        newseeds[i] = 1

        newseeds = np.multiply(newseeds,self.germMat)[:]
        #numseeds = np.add(self.numSeeds,newseeds)
        if inf:
            infSeeds = np.multiply(newseeds,self.pInf)[:]
            unInfSeeds = np.subtract(newseeds,infSeeds)[:]
            # some of seeds germinate
            germ = np.multiply(self.germMatUninf,unInfSeeds)[:]
            germ = np.add(germ,np.multiply(self.germMatInf,infSeeds)[:])[:]
        else:
            germ = np.multiply(self.germMatUninf,newseeds)[:]
        
        surv = self.perTrans[3]/(1.+(self.seedlings[0]+self.seedlings[1])*self.perTrans[0]+
                  (self.speciesEig[0]+self.speciesEig[1])*self.perTrans[2]+(self.speciesEig[2]+self.speciesEig[3]+self.speciesEig[4])*self.perTrans[1])
        
        sTos = (1-germ[i])
        
        if BFOD:
            sTos *= self.BFODmat[i]
            surv *= self.BFODmat[i]         

        compTerm = self.perSurv[i]*self.get_r(compMat,i)*(1-self.negBinP[i])/self.negBinP[i]

        if inf:
            compTerm *= self.infMat[i]

        transMat = [[sTos, compTerm],[germ[i]*surv*self.seedToAdult, self.perSurv[i]]]
        return np.max(np.abs(np.linalg.eig(transMat)[0]))

    def equilVal(self):
        # Depricated 
        i = 0
        for spec in self.species:
            equilNumInds = ((-1+self.germMatUninf[i]*self.lam[i])/self.compParams[i][i])
            print equilNumInds*self.lam[i]/(1.+equilNumInds*self.compParams[i][i]),
            i += 1
        print 
