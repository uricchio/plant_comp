import sys
import os
import numpy as np
import random
import math
from collections import defaultdict
from scipy.stats import nbinom
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.special import digamma
from scipy.optimize import minimize
from scipy.optimize import brentq

class Comp:

    def __init__(self,psd = 1e-4,rsd=1e-3,alsd = 0.01,
                 totSamps=2.*10**5,burn=2*10**4,nSamps=0,spec="AB",skip=100,
                 mydir = os.getcwd(),outf="",indId = False,nocomp=False):

        self.rsd = rsd
        self.psd = psd
        self.alsd = alsd
        self.musd = 0.02
        self.totSamps = totSamps
        self.burn = burn
        self.nSamps = nSamps
        self.spec = spec
        self.skip = skip
        self.mydir = mydir
        self.nocomp = nocomp
        self.labels = {}
        self.headers = defaultdict(dict)
        self.idn = 1
        self.indId = indId
        if os.environ.has_key('SGE_TASK_ID'):
            idn = os.environ['SGE_TASK_ID']
        if not outf:
            self.outf = os.path.join(self.mydir,"paramInfer."+str(self.idn)+".txt")
        else: 
            self.outf = outf
        self.outh = open(self.outf,'w')   
        self.data = defaultdict(dict)
        self.pred_data = defaultdict(dict)
        self.baseroot = ()
        self.al = {}
        self.mu = 0.2
        self.p0 = 1e-5
        self.r0 = 0.1
        self.al_t = {}
        self.p0_t = 1e-5
        self.r0_t = 0.1
        self.sp = "SP"
        self.eg = "EG"
        self.bd = "BD"
        self.bh = "BH"
        self.sp = "AB"
        self.sc = "Seed.estimate"
        self.idnum = "Plant.ID"
        self.area = "area"
        self.trans = "trans"
        self.treat = "treat"
        self.plot = "plot"
        self.sc_pos = -1
        self.dam = "dam"
        self.p_outs = [[] for i in range(9)]
        self.rescale_rat = []
        self.rescaled = False
        self.plot_dens = {}
        self.alldam = {}
        self.i_keys = {}

    def get_plot_dens(self,spec):
        
        for i in range(len(self.data[spec][self.sc])):
            self.plot_dens[str(self.data[spec][self.plot][i])+"."+str(int(self.data[spec][self.trans][i]))] = [self.data[spec][self.sp][i],self.data[spec][self.eg][i],self.data[spec][self.bh][i],self.data[spec][self.bd][i],self.data[spec][self.ab][i]]
            self.alldam[str(self.data[spec][self.plot][i])+"."+str(int(self.data[spec][self.trans][i]))+"."+str(self.data[spec][self.idnum][i])] = self.data[spec][self.dam][i]
            self.i_keys[str(self.data[spec][self.plot][i])+"."+str(int(self.data[spec][self.trans][i]))+"."+str(self.data[spec][self.idnum][i])] = i
  
    def max_like_p0_r0(self,d): 
        new_d = []  
        for i in range(len(d[self.sc])):
            if math.isnan(float(d[self.sc][i])):
                continue
            new_d.append(float(d[self.sc][i]))
        def myfunc(r,d):
            tot = 0
            N = len(d)
            for thing in d:
                tot += digamma(r+thing)
            return N*np.log(r/(r+np.sum(d)/N))-N*digamma(r)+tot
        myb = 1e-20
        while (np.sign(myfunc(1e-20,new_d)) == np.sign(myfunc(myb,new_d))):
             myb *= 2.
             if myb > 10**10:
                 print >> sys.stderr, "brentq failed"
                 exit()
        myroot_r = brentq(myfunc,1e-20,myb,args=(new_d))
        myroot_p = myroot_r/(myroot_r + np.sum(new_d)/(len(new_d)+0.))
     
        self.p0 = myroot_p
        self.r0 = myroot_r
        self.psd = self.p0/20.
        self.rsd = self.r0/20.
        
        self.baseroot = (myroot_r,myroot_p)

    def areaNormSeed(self,spec):
        for i in range(len(self.data[spec][self.sc])):
            self.data[spec][self.sc][i] /= (self.data[spec][self.area][i]/100.)
            self.data[spec][self.sc][i] = float(round(self.data[spec][self.sc][i]))
            #print self.data[spec][self.sc][i]

    def readPlantDiams(self,spec,pfile):
        pfh = open(pfile, 'r')
        for line in pfh:
            data = line.strip().split()
            if data[3] != spec:
                continue
            if data[2] not in self.data[spec][self.idnum]:
                continue
            if data[4] == 'NA' or data[5] == 'NA':
                continue
            if  data[1]+".0."+data[2] in self.i_keys: 
                #print self.i_keys[data[1]+".0."+data[2]]
                self.data[spec][self.area][self.i_keys[data[1]+".0."+data[2]]] = float(data[4])*float(data[5])*np.pi/4.           
            # calculate area, (a/2)*(b/2)*pi

    def prop_mu(self,spec):
        self.mu_t[spec] = np.random.normal(loc=self.mu_t[spec],scale=self.musd)   

    def prop_alpha(self,spec):
        if not self.nocomp:
            self.al_t[spec][0] = np.random.normal(loc=self.al_t[spec][0],scale=self.alsd)
            self.al_t[spec][1] = np.random.normal(loc=self.al_t[spec][1],scale=self.alsd)
            self.al_t[spec][2] = np.random.normal(loc=self.al_t[spec][2],scale=self.alsd)
            self.al_t[spec][3] = np.random.normal(loc=self.al_t[spec][3],scale=self.alsd)
            self.al_t[spec][4] = np.random.normal(loc=self.al_t[spec][4],scale=self.alsd)
            self.al_t[spec][5] = np.random.normal(loc=self.al_t[spec][5],scale=self.alsd)
        
        else:
            for i in range(0,6):
                self.al_t[spec][i] = 2e-10
            self.al_t[spec][1] = np.random.normal(loc=self.al_t[spec][1],scale=self.alsd)

    def prop_p(self):
        ret =  np.random.normal(loc=self.p0,scale=self.psd)
        if ret < 0:
            self.p0_t = 10**-7
        if ret > 1.:
            self.p0_t = 1.-10**-7
        self.p0_t = ret

    def prop_r(self):
        ret =  np.random.normal(loc=self.r0,scale=self.rsd)
        if ret < 0:
            self.r0_t =  10**-7
        self.r0_t = ret

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
                if thing != self.sc and thing != self.ab and thing != self.eg and thing != self.bh and thing != self.bd and thing != self.sp and thing != self.dam and thing != self.area:
                    if not self.indId:
                        i += 1
                        continue
                    elif self.indId and thing != self.idnum and thing != self.trans and thing != self.plot and thing != self.treat:
                        i += 1
                        continue
                if thing not in data:
                    data[thing] = []
                if locdata[i] == "NA":
                    locdata[i] = "NaN"
                if thing != self.plot:
                    try:
                        data[thing].append(float(locdata[i]))
                    except:
                        data[thing].append(locdata[i])
                else:
                    data[thing].append(locdata[i])
                i += 1
        if self.indId and self.trans not in data:
            data[self.trans] = [0. for i in range(0,len(data[self.sc]))]
        if spec not in self.data:
            self.data[spec] = data
        else:
            for tag in data:
                if tag in self.data[spec]:
                    self.data[spec][tag] = self.data[spec][tag]+data[tag][:]
                else:
                    self.data[spec][tag] = data[tag][:]
        if self.indId:
            #self.data[spec][self.idnum] = np.array([np.nan for i in range(len(self.data[spec][self.sc]))])[:]
            self.data[spec][self.area] = np.array([np.nan for i in range(len(self.data[spec][self.sc]))])[:]
        #for tag in self.data[spec]:
        #    print tag, self.data[spec][tag]
        fh.close()

    def make_seed_count_ints(self,spec):
        for i in range(len(self.data[spec][self.sc])):
            self.data[spec][self.sc][i] = round(self.data[spec][self.sc][i])

    def rep_NaN_mean(self,spec):
        time_to_break = 0
        nans = 0
        for lab in self.data[spec]: 
            for thing in self.data[spec][lab]:
                if not isinstance(thing,float):
                    time_to_break = 1
                    break
                if np.isnan(thing):
                    nans+=1
            if time_to_break == 1:
                time_to_break = 0
                continue
            if nans == len(self.data[spec][lab]):
                nans = 0
                continue
            nans = 0
            locmean= np.nanmean(self.data[spec][lab])
            for k in range(len(self.data[spec][lab])):
                if np.isnan(float(self.data[spec][lab][k])):
                    self.data[spec][lab][k] = round(locmean)

    def rem_NaN_area(self, spec):
        i = 0
        while i <len(self.data[spec][self.area]):
            if np.isnan(self.data[spec][self.area][i]):
                for tag in (self.data[spec]):
                    self.data[spec][tag].pop(i)
                continue
            i +=1

    def rep_NaN_random(self,spec):
        time_to_break = 0
        nans = 0
        for lab in self.data[spec]: 
            for thing in self.data[spec][lab]:
                if not isinstance(thing,float):
                    time_to_break = 1
                    break
                if np.isnan(thing):
                    nans+=1
            if time_to_break == 1:
                time_to_break = 0
                continue
            if nans == len(self.data[spec][lab]):
                nans = 0
                continue
            nans = 0
            for k in range(len(self.data[spec][lab])):
                if np.isnan(float(self.data[spec][lab][k])):
                    while np.isnan(float(self.data[spec][lab][k])):
                        self.data[spec][lab][k] = np.random.choice(self.data[spec][lab])

    def init_params(self,spec):
        self.max_like_p0_r0(self.data[spec])
        if not self.nocomp:
            self.al[spec] = np.add(np.zeros(6),[0.1*np.random.random() for i in range(0,6)])
        else:
            self.al[spec] = np.zeros(6)

    def map_labels(self,labels):
        self.sp  = labels[0]
        self.eg  = labels[1]
        self.bh  = labels[2]
        self.bd  = labels[3]
        self.ab  = labels[4]
        self.dam  = labels[5]
        self.sc = labels[6]

    def get_r(self,spec):
        
        if not self.rescaled:
            self.rescale(spec)

        D0 = self.data[spec][self.sp]
        D1 = self.data[spec][self.eg]  
        D2 = self.data[spec][self.bh]
        D3 = self.data[spec][self.bd]
        D4 = self.data[spec][self.ab]
        D5 = self.data[spec][self.dam]
        r0 =  self.r0
        al0 = self.al[spec][0]/self.rescale_rat[0]
        al1 = self.al[spec][1]/self.rescale_rat[1]
        al2 = self.al[spec][2]/self.rescale_rat[2]
        al3 = self.al[spec][3]/self.rescale_rat[3]
        al4 = self.al[spec][4]/self.rescale_rat[4]
        al5 = self.al[spec][5]/self.rescale_rat[5]

        ret = np.zeros(len(D0))
        for i in range(len(D0)):
            #ret[i] = r0/(1.+ D0[i]*al0 +D1[i]*al1 +D2[i]*al2 +D3[i]*al3 +D4[i]*al4 +D5[i]*al5)
            ret[i] = r0*math.exp(-1.*(D0[i]*al0 +D1[i]*al1 +D2[i]*al2 +D3[i]*al3 +D4[i]*al4 +D5[i]*al5))
        return ret

    def transform_densities(self,spec):

        D0 = self.data[spec][self.sp]
        D1 = self.data[spec][self.eg]
        D2 = self.data[spec][self.bh]
        D3 = self.data[spec][self.bd]
        D4 = self.data[spec][self.ab]
    
        Dper = np.add(D0,D1)[:]
        Dann = np.add(np.add(D2,D3)[:],D4)[:]
        Dperper = np.multiply(Dper,Dper)[:]
        Dannann = np.multiply(Dann,Dann)[:]
        Dannper = np.multiply(Dper,Dann)[:]

        self.data[spec][self.sp] = Dper[:]
        self.data[spec][self.eg] = Dann[:]
        self.data[spec][self.bh] = Dperper[:]
        self.data[spec][self.bd] = Dannann[:]
        self.data[spec][self.ab] = Dannper[:]
    
    def get_r_t(self,spec):
        
        if not self.rescaled:
            self.rescale(spec)
        
        D0 = self.data[spec][self.sp]
        D1 = self.data[spec][self.eg]  
        D2 = self.data[spec][self.bh]
        D3 = self.data[spec][self.bd]
        D4 = self.data[spec][self.ab]
        D5 = self.data[spec][self.dam]
        r0 =  self.r0_t
        al0 = self.al_t[spec][0]/self.rescale_rat[0]
        al1 = self.al_t[spec][1]/self.rescale_rat[1]
        al2 = self.al_t[spec][2]/self.rescale_rat[2]
        al3 = self.al_t[spec][3]/self.rescale_rat[3]
        al4 = self.al_t[spec][4]/self.rescale_rat[4]
        al5 = self.al_t[spec][5]/self.rescale_rat[5]

        ret = np.zeros(len(D0))
        for i in range(len(D0)):
            #ret[i] = r0/(1.+ D0[i]*al0 +D1[i]*al1 +D2[i]*al2 +D3[i]*al3 +D4[i]*al4 +D5[i]*al5)
            ret[i] = r0*math.exp(-1.*(D0[i]*al0 +D1[i]*al1 +D2[i]*al2 +D3[i]*al3 +D4[i]*al4 +D5[i]*al5))
        return ret

    def sim_seed(self,spec):

        if not self.rescaled:
            self.rescale(spec)

        if not self.baseroot:
            self.max_like_p0_r0(self.data[spec][self.sc])
        # model-based simulation of seed output
 
        # first get al values, noting that not all combinations are possible (some result in negative mean)
        testarr = []
        while 1:
            #self.al[spec][0] = np.arange(-0.1,0.1,0.01)[np.random.choice(xrange(20))]
            #self.al[spec][1] = np.arange(-0.1,0.1,0.01)[np.random.choice(xrange(20))]
            #self.al[spec][2] = np.arange(-0.001,0.01,0.0001)[np.random.choice(xrange(110))]
            #self.al[spec][3] = np.arange(-0.001,0.01,0.0001)[np.random.choice(xrange(110))]
            #self.al[spec][4] = np.arange(-0.05,0.05,0.005)[np.random.choice(xrange(20))]
            #self.al[spec][5] = np.arange(-50,50,5)[np.random.choice(xrange(20))]
            myrange = np.arange(0.01,1.01,0.01)
            self.al[spec][0] = myrange[np.random.choice(xrange(100))]
            self.al[spec][1] = myrange[np.random.choice(xrange(100))]
            self.al[spec][2] = myrange[np.random.choice(xrange(100))]
            self.al[spec][3] = myrange[np.random.choice(xrange(100))]
            self.al[spec][4] = myrange[np.random.choice(xrange(100))]
            self.al[spec][5] = myrange[np.random.choice(xrange(100))]

            testarr = self.get_r(spec)
                              
            br = 0

            for thing in testarr:
                if thing <= 0.:
                    br += 1
                    break
            if br == 0:
                break
        self.modelrvs(spec)

    def modelrvs(self,spec): 
        # simulate seed  data under model
        r_loc = self.get_r(spec)
        self.data[spec][self.sc] =  nbinom.rvs(n=r_loc,p=self.p0)
       
    #def logp0(sc,r,p):
    #    if sc == 0:
    #        return q + (1-q)*nbinom.logpmf(sc,n=r,p=0)
    #    return (1-q)*nbinom.logpmf(sc,n=r,p=0)

    def ll(self,spec):
        r_loc = self.get_r(spec)
        for thing in r_loc:
            if thing < 0:
                return -(10**50)
        if spec in ["SP","EG"]:
            ret = np.sum(nbinom.logpmf(self.data[spec][self.sc],n=np.multiply(r_loc,np.sqrt(self.data[spec][self.area])), p=self.p0))
        else:
            ret = np.sum(nbinom.logpmf(self.data[spec][self.sc],n=r_loc, p=self.p0))
        mylen = len(self.data[spec][self.sc])
        for val in self.al[spec]:
            #ret+=mylen*(norm.logpdf(val ,loc=0,scale=10))
            ret+=mylen*(gamma.logpdf(val, a=1,scale=300))
        #ret+=mylen*(norm.logpdf(self.al[spec][5] ,loc=0,scale=10))
        return ret
    
    def llNoPrior(self,spec):
        r_loc = self.get_r(spec)
        for thing in r_loc:
            if thing < 0:
                return -(10**50)
        if spec in ["SP","EG"]:
            ret = np.sum(nbinom.logpmf(self.data[spec][self.sc],n=np.multiply(r_loc,np.sqrt(self.data[spec][self.area])), p=self.p0))
        else: 
            ret = np.sum(nbinom.logpmf(self.data[spec][self.sc],n=r_loc, p=self.p0))
        #mylen = len(self.data[spec][self.sc])
        #for val in self.al[spec]:
            #ret+=mylen*(norm.logpdf(val ,loc=0,scale=10))
        #    ret+=mylen*(gamma.logpdf(val, a=1,scale=100))
        #ret+=mylen*(norm.logpdf(self.al[spec][5] ,loc=0,scale=10))
        return ret

    def ll_t(self,spec):
        r_loc = self.get_r_t(spec)
        for thing in r_loc:
            if thing < 0:
                return -(10**50)
        if spec in ["SP","EG"]:
            ret = np.sum(nbinom.logpmf(self.data[spec][self.sc],n=np.multiply(r_loc,np.sqrt(self.data[spec][self.area])), p=self.p0_t))
        else:
            ret = np.sum(nbinom.logpmf(self.data[spec][self.sc],n=r_loc, p=self.p0_t))
        #ret = 0.
        #i = 0
        #for sc in self.data[spec][self.sc]:
        #    ret += logp0(sc,r_loc[i],self.p0,self.q0)
        #    i += 1
        mylen = len(self.data[spec][self.sc])
        for val in self.al_t[spec]:
            #ret+=mylen*(norm.logpdf(val ,loc=0,scale=10))
            ret+=mylen*(gamma.logpdf(val, a=1,scale=300))
        #ret+=mylen*(norm.logpdf(self.al[spec][5] ,loc=0,scale=10))
        return ret

    def write_params(self,spec,init=False):
        outh = open(self.outf,'a')
        if init:
            outh.write('# ')
        outh.write(str(self.r0)+' '+str(self.p0)+' ')
        tags = [self.sp,self.eg,self.bh,self.bd,self.ab,self.dam]
        for thing in self.al[spec]:
            outh.write(str(thing)+' ')
        i = 0
        for thing in self.al[spec]:
            outh.write(str(thing/self.rescale_rat[i])+' ')
            i += 1
        ll0 = self.ll(spec)
        outh.write(str(ll0)+'\n')
        
    def scaleDens(self,spec,tag):
        for i in range(len(self.data[spec][tag])):
            #if self.data[spec][tag][i] == 0.0:
            #    continue
            self.data[spec][tag][i] = self.data[spec][tag][i]**2
 
    def burn_in(self,spec):

        if not self.rescaled:
            self.rescale(spec)
        
        #for seed in self.data[spec][self.sc]:
        #    print seed
        #exit()     

        accept=np.zeros(self.burn)

        param_counter = 0

        ll_cur = self.ll(spec)
        if spec not in self.al_t:
            self.al_t[spec] = np.zeros(len(self.al[spec]))

        nSamps = 0
        ll_next = 0   
        diff = 1e-20
        while nSamps < self.burn: 

            self.p0_t = self.p0+diff
            self.r0_t = self.r0+diff
            for i in range(len(self.al[spec])):
                self.al_t[spec][i] = self.al[spec][i]+diff
            
            if param_counter == 1:
                self.prop_p()
            if param_counter == 2:
                self.prop_r()
            if param_counter == 0:
                self.prop_alpha(spec)

            ll_next = self.ll_t(spec)

            if ll_next > ll_cur:
                self.p0 = self.p0_t-diff
                self.r0 = self.r0_t-diff
                for i in range(len(self.al[spec])):
                    self.al[spec][i] = self.al_t[spec][i]-diff
                ll_cur = ll_next
                accept[nSamps]=1
            elif ll_next < ll_cur: 
                rat = np.exp(ll_next-ll_cur)
                if  np.random.random() < rat:
                    ll_cur = ll_next
                    accept[nSamps]=1 
                    self.p0 = self.p0_t-diff
                    self.r0 = self.r0_t-diff
                    for i in range(len(self.al[spec])):
                        self.al[spec][i] = self.al_t[spec][i]-diff
            nSamps+=1
        
            if nSamps > 1 and nSamps % 200 == 0:
                #print ll_cur, np.divide(self.al[spec],self.rescale_rat), self.p0, self.r0, self.rsd, self.psd
                print ll_cur, self.al[spec], self.p0, self.r0, self.rsd, self.psd, np.sum(accept[(nSamps-200):nSamps])/200.
                
                #print self.alsd, self.psd, self.rsd, ll_cur
                if param_counter == 0:
                    if np.sum(accept[(nSamps-200):nSamps])/200. < 0.35:
                        self.alsd /= 1.1
                    elif np.sum(accept[(nSamps-200):nSamps])/200. >= 0.35:
                        self.alsd*=1.1
                    param_counter += 1
                    continue
                elif param_counter == 1:
                    if np.sum(accept[(nSamps-200):nSamps])/200. < 0.35:
                        self.psd /= 1.1
                    elif np.sum(accept[(nSamps-200):nSamps])/200. >= 0.35:
                        self.psd*=1.1
                    param_counter += 1
                    continue
                elif param_counter == 2:
                    if np.sum(accept[(nSamps-200):nSamps])/200. < 0.35:
                        self.rsd /= 1.1
                    elif np.sum(accept[(nSamps-200):nSamps])/200. >= 0.35:
                        self.rsd *= 1.1
                    param_counter = 0
                    continue

        return np.sum(accept[(nSamps-200):nSamps])/200.

    def mcmc(self,spec):

        #if rescale:
        #    self.rescale(spec)
        
        ll_cur = self.ll(spec)
        if spec not in self.al_t:
            self.al_t[spec] = np.zeros(len(self.al[spec]))

        nSamps = 0
        ll_next = 0   
        diff = 1e-10
        while nSamps < self.totSamps: 

            self.p0_t = self.p0+diff
            self.r0_t = self.r0+diff
            for i in range(len(self.al[spec])):
                self.al_t[spec][i] = self.al[spec][i]+diff
            
            self.prop_p()
            self.prop_r()
            self.prop_alpha(spec)

            ll_next = self.ll_t(spec)

            if ll_next > ll_cur:
                self.p0 = self.p0_t-diff
                self.r0 = self.r0_t-diff
                for i in range(len(self.al[spec])):
                    self.al[spec][i] = self.al_t[spec][i]-diff
                ll_cur = ll_next
            elif ll_next < ll_cur: 
                rat = np.exp(ll_next-ll_cur)
                if  np.random.random() < rat:
                    ll_cur = ll_next
                    self.p0 = self.p0_t-diff
                    self.r0 = self.r0_t-diff
                    for i in range(len(self.al[spec])):
                        self.al[spec][i] = self.al_t[spec][i]-diff
            nSamps+=1
        
            if nSamps > 1 and nSamps % self.skip == 0:
                self.write_params(spec)       
                for i in range(6):
                    self.p_outs[i].append(self.al[spec][i])
                self.p_outs[6].append(self.r0)
                self.p_outs[7].append(self.p0)
                self.p_outs[8].append(self.llNoPrior(spec))

    def rescale(self,spec):
 
        if self.rescaled:
            return
        ret = []
        tags = [self.sp,self.eg,self.bh,self.bd,self.ab,self.dam]
        for tag in tags:
            mysd = np.var(self.data[spec][tag])**0.5
            #self.data[spec][tag] = np.divide(self.data[spec][tag],mysd)
            ret.append(mysd)
        self.rescale_rat = ret
        self.rescaled = True

    def compute_pred(self,spec,params,p0,r0):

        tags = [self.sp,self.eg,self.bh,self.bd,self.ab,self.dam]

        denoms = []
 
        for ind in range(len(self.data[spec][tags[0]])):
            denom = 1.
            for i in range(len(params)):
                denom+=self.data[spec][tags[i]][ind]*params[i]/self.rescale_rat[i]         

            denoms.append((r0*(1-p0)/p0)*1/denom)
  
        return denoms           
        

    def compute_pred_split(self,spec):

        tags = [self.sp,self.eg,self.bh,self.bd,self.ab,self.dam]

        params = []

        for arr in self.p_outs:
            params.append(np.mean(arr))

        denoms = []
        r_loc = []

        for ind in range(len(self.pred_data[spec][tags[0]])):
            denom = 1.
            for i in range(6):
                denom+=self.pred_data[spec][tags[i]][ind]*params[i]/self.rescale_rat[i]
            r_loc.append(params[6]/denom)
            #if denoms > 1:
            #    denom = 1 
            denoms.append((params[6]*(1-params[7])/params[7])*1./denom)
                        
        lls = nbinom.cdf(np.median(self.pred_data[spec][self.sc]),n=r_loc,p=params[7])
        ll0 = nbinom.cdf(np.median(self.pred_data[spec][self.sc]),n=self.baseroot[0],p=self.baseroot[1])
  
        llall = np.sum(nbinom.logpmf(self.pred_data[spec][self.sc],n=r_loc,p=params[7]))
        llallnull = np.sum(nbinom.logpmf(self.pred_data[spec][self.sc],n=self.baseroot[0],p=self.baseroot[1]))

        return (denoms,lls,ll0,np.median(self.pred_data[spec][self.sc]),llall,llallnull)
 
    def return_max_ll(self,spec,writefile):
    
        ll0 =  np.sum(nbinom.logpmf(self.data[spec][self.sc],n=self.baseroot[0],p=self.baseroot[1]))
        ll1 = max(self.p_outs[8])
        
        wh = open(writefile,'w')

        wh.write(str(ll1)+' '+str(ll0)+'\n')

        wh.close()

    def compute_model_prob(self,spec,writefile):

        tags = [self.sp,self.eg,self.bh,self.bd,self.ab,self.dam]

        params = []

        for arr in self.p_outs:
            params.append(np.mean(arr))

        r_loc = []

        for ind in range(len(self.data[spec][tags[0]])):
            denom = 1.
            for i in range(6):
                denom+=self.data[spec][tags[i]][ind]*params[i]/self.rescale_rat[i]
            r_loc.append(params[6]/denom)
            #if denoms > 1:
            #    denom = 1 

        llall = np.sum(nbinom.logpmf(self.data[spec][self.sc],n=r_loc,p=params[7]))
        llallnull = np.sum(nbinom.logpmf(self.data[spec][self.sc],n=self.baseroot[0],p=self.baseroot[1]))
         
        wh = open(writefile,'w')
        
        wh.write(str(llall)+' '+str(llallnull)+'\n')
   
        wh.close()

    def split_data(self,spec):
    
        split_point = int(round((len(self.data[spec][self.sp])+0.)/2.))

        tags = [self.sc,self.sp,self.eg,self.bh,self.bd,self.ab,self.dam]     

        for tag in tags:
            self.pred_data[spec][tag] = []

        while len(self.data[spec][self.sp]) > split_point:
            popval = np.random.randint(len(self.data[spec][self.sp]))
            for tag in tags:
                self.pred_data[spec][tag].append(self.data[spec][tag].pop(popval))

    def print_labs(self,spec,labs):
    
        for i in xrange(len(self.data[spec][labs[0]])):
            for lab in labs:
                print self.data[spec][lab][i],
            print
           
