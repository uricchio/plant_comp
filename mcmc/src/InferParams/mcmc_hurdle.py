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

    def __init__(self,psd = 1e-4,rsd=1e-3,qsd=1e-3,alsd = 0.01,
                 totSamps=2.*10**5,burn=2*10**4,nSamps=0,spec="AB",skip=100,
                 mydir = os.getcwd(),outf=""):

        self.rsd = rsd
        self.psd = psd
        self.qsd = qsd
        self.alsd = alsd
        self.totSamps = totSamps
        self.burn = burn
        self.nSamps = nSamps
        self.spec = spec
        self.skip = skip
        self.mydir = mydir
        self.labels = {}
        self.headers = defaultdict(dict)
        self.idn = 1
        if os.environ.has_key('SGE_TASK_ID'):
            idn = os.environ['SGE_TASK_ID']
        if not outf:
            self.outf = os.path.join(self.mydir,"paramInfer."+str(self.idn)+".txt")
        else: 
            self.outf = outf
        self.outh = open(self.outf,'w')   
        self.data = defaultdict(dict)
        self.sim_data = defaultdict(dict)
        self.pred_data = defaultdict(dict)
        self.baseroot = ()
        self.al = {}
        self.p0 = 1e-5
        self.r0 = 0.1
        self.q0 = 0.01
        self.sp = "SP"
        self.eg = "EG"
        self.bd = "BD"
        self.bh = "BH"
        self.sp = "AB"
        self.dam = "dam"
        self.p_outs = [[] for i in range(9)]
        self.rescale_rat = []
        self.rescaled = False
        self.max_q0 = 0.01
        self.q_skip = False

    def max_like_p0_r0(self,d): 
        new_d = []  
        for i in range(len(d[self.sc])):
            if math.isnan(float(d[self.sc][i])):
                continue
            if float(d[self.sc][i]) < 1:
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

    def prop_alpha(self,spec):

        alret = np.zeros(6)

        alret[0] = np.random.normal(loc=self.al[spec][0],scale=self.alsd)
        alret[1] = np.random.normal(loc=self.al[spec][1],scale=self.alsd)
        alret[2] = np.random.normal(loc=self.al[spec][2],scale=self.alsd)
        alret[3] = np.random.normal(loc=self.al[spec][3],scale=self.alsd)
        alret[4] = np.random.normal(loc=self.al[spec][4],scale=self.alsd)
        alret[5] = np.random.normal(loc=self.al[spec][5],scale=self.alsd)

        return alret

    def prop_p(self):
        ret =  np.random.normal(loc=self.p0,scale=self.psd)
        if ret < 0:
            return 10**-7
        if ret > 1.:
            return 1.-10**-7
        return ret

    def prop_r(self):
        ret =  np.random.normal(loc=self.r0,scale=self.rsd)
        if ret < 0:
            return  10**-7
        return ret

    def prop_q(self):
        ret =np.random.normal(loc=self.q0,scale=self.qsd)
        if ret < 0:
            return  10**-7
        if ret > self.max_q0:
            return self.max_q0
        return ret

    def get_headers(self,f):
        fh = open(f,'r') 
        labels = []
        for line in fh:
            line = line.strip().split(',')
           
            for thing in line:
                if "\"" in thing:
                    thing = thing[1:-1]
                if f not in self.labels:
                    self.labels[f] = []
                self.labels[f].append(thing)
                if f not in self.headers:
                    self.headers[f][thing] = []
                self.headers[f][thing] = []
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
            for k in range(len(locdata)):
                if "\"" in locdata[k]:
                    locdata[k] = locdata[k][1:-1] 
            i = 0
            notspec = 0
            catch_nan = 0
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
                    self.data[spec][tag] = self.data[spec][tag]+data[tag]
                else:
                    self.data[spec][tag] = self.data[tag]
        for tag in self.data[spec]:
            if self.data[spec][tag]:
                self.sim_data[spec][tag] = self.data[spec][tag][:]
        
        #for thing in self.data[spec]["Seed.estimate"]:
        #    print thing
        fh.close()

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

    def rep_NaN_random(self,spec):
        time_to_break = 0
        nans = 0
        for lab in self.sim_data[spec]: 
            for thing in self.sim_data[spec][lab]:
                if not isinstance(thing,float):
                    time_to_break = 1
                    break
                if np.isnan(thing):
                    nans+=1
            if time_to_break == 1:
                time_to_break = 0
                continue
            if nans == len(self.sim_data[spec][lab]):
                nans = 0
                continue
            nans = 0
            for k in range(len(self.sim_data[spec][lab])):
                if np.isnan(float(self.sim_data[spec][lab][k])):
                    while np.isnan(float(self.sim_data[spec][lab][k])):
                        self.sim_data[spec][lab][k] = np.random.choice(self.sim_data[spec][lab])

    def init_params(self,spec):
        self.max_like_p0_r0(self.data[spec])
        self.al[spec] = np.zeros(6) 
        tot = 0.
        f0 = 0.
        for thing in self.data[spec][self.sc]:
            if thing == 0:
                f0 += 1
            tot += 1
        q0 = f0/tot
        self.q0 = q0
        #print q0
        self.max_q0 = q0
        if f0 == 0.:
            self.q_skip = True
        else:
           self.q_skip = False

    def map_labels(self,labels):
        self.sp  = labels[0]
        self.eg  = labels[1]
        self.bh  = labels[2]
        self.bd  = labels[3]
        self.ab  = labels[4]
        self.dam  = labels[5]
        self.sc = labels[6]

    def get_r(self,spec,sim=False):
        
        if not self.rescaled:
            self.rescale(spec)

        if sim == False:
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
                ret[i] = r0/(1.+ D0[i]*al0 +D1[i]*al1 +D2[i]*al2 +D3[i]*al3 +D4[i]*al4 +D5[i]*al5)
            return ret

        else:
            D0 = self.sim_data[spec][self.sp]
            D1 = self.sim_data[spec][self.eg]
            D2 = self.sim_data[spec][self.bh]
            D3 = self.sim_data[spec][self.bd]
            D4 = self.sim_data[spec][self.ab]
            D5 = self.sim_data[spec][self.dam]
            r0 =  self.r0
            al0 = self.al[spec][0]/self.rescale_rat[0]
            al1 = self.al[spec][1]/self.rescale_rat[1]
            al2 = self.al[spec][2]/self.rescale_rat[2]
            al3 = self.al[spec][3]/self.rescale_rat[3]
            al4 = self.al[spec][4]/self.rescale_rat[4]
            al5 = self.al[spec][5]/self.rescale_rat[5]

            ret = np.zeros(len(D0))
            for i in range(len(D0)):
                ret[i] = r0/(1.+ D0[i]*al0 +D1[i]*al1 +D2[i]*al2 +D3[i]*al3 +D4[i]*al4 +D5[i]*al5)
            return ret



    def get_r_t(self,spec,rt,alt):
        
        if not self.rescaled:
            self.rescale(spec)
        
        D0 = self.data[spec][self.sp]
        D1 = self.data[spec][self.eg]  
        D2 = self.data[spec][self.bh]
        D3 = self.data[spec][self.bd]
        D4 = self.data[spec][self.ab]
        D5 = self.data[spec][self.dam]
        r0 =  rt
        al0 = alt[0]/self.rescale_rat[0]
        al1 = alt[1]/self.rescale_rat[1]
        al2 = alt[2]/self.rescale_rat[2]
        al3 = alt[3]/self.rescale_rat[3]
        al4 = alt[4]/self.rescale_rat[4]
        al5 = alt[5]/self.rescale_rat[5]

        ret = np.zeros(len(D0))
        for i in range(len(D0)):
            ret[i] = r0/(1.+ D0[i]*al0 +D1[i]*al1 +D2[i]*al2 +D3[i]*al3 +D4[i]*al4 +D5[i]*al5)
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
            myrange = np.arange(-0.5,0.5,0.01)
            self.al[spec][0] = myrange[np.random.choice(xrange(100))]
            self.al[spec][1] = myrange[np.random.choice(xrange(100))]
            self.al[spec][2] = myrange[np.random.choice(xrange(100))]
            self.al[spec][3] = myrange[np.random.choice(xrange(100))]
            self.al[spec][4] = myrange[np.random.choice(xrange(100))]
            self.al[spec][5] = myrange[np.random.choice(xrange(100))]

            testarr = self.get_r(spec,sim=True)
                              
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
        r_loc = self.get_r(spec,sim=True)
        for i in range(len(self.sim_data[spec][self.sc])):
            if np.random.random() > self.q0:
                self.data[spec][self.sc][i] =  nbinom.rvs(n=r_loc[i],p=self.p0)
            else:
                self.data[spec][self.sc][i] = 0    

    def ll(self,spec):
        def logp0(sc,r,p,q):
            if sc == 0:
                return np.log(q + (1-q)*nbinom.pmf(sc,n=r,p=p))
            return math.log(1-q) + nbinom.logpmf(sc,n=r,p=p)
        r_loc = self.get_r(spec)
        for thing in r_loc:
            if thing < 0:
                return -(10**50)
        #ret = np.sum(nbinom.logpmf(self.data[spec][self.sc],n=r_loc, p=self.p0))
        ret = 0.
        i = 0
        if self.q_skip:
            ret += np.sum(nbinom.logpmf(self.data[spec][self.sc],n=r_loc,p=self.p0))
        else:
            for sc in self.data[spec][self.sc]:
                ret += logp0(sc,r_loc[i],self.p0,self.q0)
                i += 1
        mylen = len(self.data[spec][self.sc])
        for val in self.al[spec]:
            ret+=mylen*(norm.logpdf(val ,loc=0,scale=10))
        return ret
    
    def ll_t(self,spec,pt,rt,qt,alt):
        def logp0(sc,r,p,q):
            if sc == 0:
                return np.log(q + (1-q)*nbinom.pmf(sc,n=r,p=p))
            return math.log(1-q) + nbinom.logpmf(sc,n=r,p=p)
        r_loc = self.get_r_t(spec,rt,alt)
        for thing in r_loc:
            if thing < 0:
                return -(10**50)
        #ret = np.sum(nbinom.logpmf(self.data[spec][self.sc],n=r_loc, p=self.p0_t))
        ret = 0.
        i = 0
        if self.q_skip:
            ret += np.sum(nbinom.logpmf(self.data[spec][self.sc],n=r_loc,p=pt))
        else:
            for sc in self.data[spec][self.sc]:
                ret += logp0(sc,r_loc[i],pt,qt)
                i += 1
        mylen = len(self.data[spec][self.sc])
        for val in alt:
            ret+=mylen*(norm.logpdf(val ,loc=0,scale=10))
        return ret

    def write_params(self,spec,init=False):
        outh = open(self.outf,'a')
        if init:
            outh.write('# ')
        outh.write(str(self.r0)+' '+str(self.p0)+' '+str(self.q0)+' ')
        tags = [self.sp,self.eg,self.bh,self.bd,self.ab,self.dam]
        for thing in self.al[spec]:
            outh.write(str(thing)+' ')
        i = 0
        for thing in self.al[spec]:
            outh.write(str(thing/self.rescale_rat[i])+' ')
            i += 1
        ll0 = self.ll(spec)
        outh.write(str(ll0)+'\n')
        
    def burn_in(self,spec):

        if not self.rescaled:
            self.rescale(spec)
        
        accept=np.zeros(self.burn)

        param_counter = 0

        ll_cur = self.ll(spec)

        nSamps = 0
        ll_next = 0   
        diff = 1e-20

        pt = self.p0+diff
        rt = self.r0+diff
        qt = self.q0+diff
        alt = self.al[spec][:]

        pt -= diff
        rt -= diff
        qt -= diff

        while nSamps < self.burn: 

            #if param_counter == 0:
            pt = self.prop_p()
            #if param_counter == 1:
            rt = self.prop_r()
            #if param_counter == 2:
            alt = self.prop_alpha(spec)
            if not self.q_skip:
                qt = self.prop_q()

            ll_next = self.ll_t(spec,pt,rt,qt,alt)

            #print ll_cur, ll_next

            if ll_next > ll_cur:
                self.p0 = pt 
                self.r0 = rt 
                if not self.q_skip:
                    self.q0 = qt 
                self.al[spec] = alt[:]
                ll_cur = ll_next
                accept[nSamps]=1
            elif ll_next < ll_cur: 
                rat = np.exp(ll_next-ll_cur)
                if  np.random.random() < rat:
                    ll_cur = ll_next
                    accept[nSamps]=1 
                    self.p0 = pt 
                    self.r0 = rt 
                    if not self.q_skip:
                        self.q0 = qt 
                    self.al[spec] = alt[:] 
            #print accept[nSamps]
            nSamps+=1
        
            if nSamps > 1 and nSamps % 200 == 0:
                
                a_ratio = np.sum(accept[(nSamps-200):nSamps])/200.

                #print ll_cur, self.al[spec], self.p0, self.r0, self.q0, self.rsd, self.psd, self.qsd, a_ratio
                #print ll_cur, np.divide(self.al[spec],self.rescale_rat), self.p0, self.r0, self.rsd, self.psd
                print ll_cur, self.al[spec], self.p0, self.r0, self.q0, self.rsd, self.psd, self.qsd, self.alsd, a_ratio
                
                #print self.alsd, self.psd, self.rsd, ll_cur
                #if param_counter == 2:
                if a_ratio < 0.35:
                    self.alsd /= 1.1
                elif a_ratio >= 0.35:
                    self.alsd*=1.1
                #if self.q_skip:
                #    param_counter = 0
                #else:
                #    param_counter += 1
                #continue
                #elif param_counter == 0:
                if a_ratio < 0.35:
                    self.psd /= 1.1
                elif a_ratio >= 0.35:
                    self.psd*=1.1
                #param_counter += 1
                #continue
                #elif param_counter == 3:
                if not self.q_skip:
                    if a_ratio < 0.35:
                        self.qsd /= 1.1
                    elif a_ratio >= 0.35:
                        self.qsd*=1.1
                #param_counter =0
                #continue
                #elif param_counter == 1:
                if a_ratio < 0.35:
                    self.rsd /= 1.1
                elif a_ratio >= 0.35:
                    self.rsd *= 1.1
                #param_counter += 1
                #continue

        return np.sum(accept[(nSamps-200):nSamps])/200.

    def mcmc(self,spec):

        #if rescale:
        #    self.rescale(spec)
        
        ll_cur = self.ll(spec)

        nSamps = 0
        ll_next = 0   
        diff = 1e-20

        pt = self.p0+diff
        rt = self.r0+diff
        qt = self.q0+diff
        alt = self.al[spec][:]

        pt -= diff
        rt -= diff
        qt -= diff

        while nSamps < self.totSamps: 

            pt = self.prop_p()
            rt = self.prop_r()
            if not self.q_skip:
                qt = self.prop_q()
            alt = self.prop_alpha(spec)

            ll_next = self.ll_t(spec,pt,rt,qt,alt)

            #print ll_cur, ll_next

            if ll_next > ll_cur:
                self.p0 = pt 
                self.r0 = rt 
                if not self.q_skip:
                    self.q0 = qt 
                self.al[spec] = alt 
                ll_cur = ll_next
            elif ll_next < ll_cur: 
                rat = np.exp(ll_next-ll_cur)
                if  np.random.random() < rat:
                    ll_cur = ll_next
                    self.p0 = pt
                    self.r0 = rt
                    if not self.q_skip:
                        self.q0 = qt
                    self.al[spec] = alt[:]
            nSamps+=1


            if nSamps > 1 and nSamps % self.skip == 0:
                self.write_params(spec)       
                for i in range(6):
                    self.p_outs[i].append(self.al[spec][i])
                self.p_outs[6].append(self.r0)
                self.p_outs[7].append(self.p0)
                self.p_outs[8].append(self.q0)

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

    def compute_pred(self,spec,params,p0,r0,q0):

        tags = [self.sp,self.eg,self.bh,self.bd,self.ab,self.dam]

        denoms = []
 
        for ind in range(len(self.data[spec][tags[0]])):
            denom = 1.
            for i in range(len(params)):
                denom+= (1-self.q0)*self.data[spec][tags[i]][ind]*params[i]/self.rescale_rat[i]

            denoms.append((r0*(1-p0)/p0)*1/denom)
  
        return denoms           
        

    def compute_pred_split(self,spec):

        tags = [self.sp,self.eg,self.bh,self.bd,self.ab,self.dam]

        params = []

        for arr in self.p_outs:
            params.append(np.mean(arr))

        denoms = []

        for ind in range(len(self.pred_data[spec][tags[0]])):
            denom = 1.
            for i in range(6):
                denom+=(1-params[8])*self.pred_data[spec][tags[i]][ind]*params[i]/self.rescale_rat[i]
            #if denoms > 1:
            #    denom = 1 
            denoms.append((params[6]*(1-params[7])/params[7])*1./denom)
            
        return denoms

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
           
