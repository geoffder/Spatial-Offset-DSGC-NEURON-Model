from neuron import h, gui
# science/math libraries
import numpy as np
import pandas as pd
import scipy.stats as st  # for probabilistic distributions
import matplotlib.pyplot as plt
# general libraries
import platform
# local imports
from modelUtils import findOrigin, rotate, findSpikes

plat = platform.system()
if plat == 'Linux':
    user = 'mouse'
    h('load_file("RGCmodel.hoc")')
    basest = '/home/'+user+'/Desktop/NEURONoutput/'
else:
    user = 'geoff'
    h('load_file("RGCmodel.hoc")')
    basest = 'C:\\Users\\'+user+'\\NEURONoutput\\'

# workaround for GUI bug in windows that prevents graphs drawing during run
if plat == 'Windows':
    h('''
        proc advance() {
            fadvance()
            nrnpython("")
        }
    ''')

# ------------ MODEL RUN PARAMETERS -----------------------------
h.tstop = 750  # (ms)
h.steps_per_ms = 10  # [10 = 10kHz]
h.dt = .1  # (ms) [.1 = 10kHz]
h.v_init = -60
h.celsius = 36.9
# -----------------------------------------------------------

h('objref RGC')
h('RGC = new DSGC(0,0)')
soma = h.RGC.soma
allDends = h.RGC.dend

runLabel = ''  # empty unless repeated run function in use
trial = ''

threshold = -45
termSynOnly = 1  # synapses only on terminal branches

# ------------------MEMBRANE PROPERTIES ---------------------------
# settings
dendSegs = 1
segStep = 1/(dendSegs*2)

activeSOMA = 1
activeDEND = 1  # set primaries and branches to active
activeSYN = 1  # only synapse branches are active (^leave primaries)
TTX = 0  # set Nav to 0 (turns on dendritic recordings)
dendPas = 0  # pas mechanism rather than HH without Na
vcPas = 0  # set passive properties for voltage-clamp

# membrane noise
dend_nzFactor = 0  # default NF_HHst = 1 (try with .5)
soma_nzFactor = .25

# soma active properties
somaNa = .15  # (S/cm2)
somaK = .07  # (S/cm2)
soma_gleak_hh = .0001667  # (S/cm2)
soma_eleak_hh = -60.0  # (mV)
soma_gleak_pas = .0001667  # (S/cm2)
soma_eleak_pas = -60  # (mV)

# dend compartment active properties
dendNa = .03  # (S/cm2)
dendK = .035  # (S/cm2)
dend_gleak_hh = 0.0001667  # (S/cm2)
dend_eleak_hh = -60.0  # (mV)
dend_gleak_pas = .0001667  # (S/cm2)
dend_eleak_pas = -60  # (mV)

# primary dend compartment active properties
primNa = .15  # (S/cm2)
primK = .07  # (S/cm2)
prim_gleak_hh = 0.0001667  # (S/cm2)
prim_eleak_hh = -60.0  # (mV)
prim_gleak_pas = .0001667  # (S/cm2)
prim_eleak_pas = -60  # (mV)

if TTX:
    somaNa = 0
    hillNa = 0
    thinNa = 0
    dendNa = 0
    primNa = 0

if vcPas:
    activeSOMA = 0
    activeDEND = 0
    activeSYN = 0
    dendPas = 1
    soma_gleak_pas = 0
    dend_gleak_pas = 0
    prim_gleak_pas = 0
# -----------------------------------------------------------

# ------------------ NMDA SETTINGS ---------------------------
# NMDA settings
sensNMDA = 1  # whether NMDA is high sensitivity (low C50 etc)
alonNMDA = 1  # use Alon's NMDA voltage-conductance function
NMDAmode = 1  # 1: voltage dependent, 0: voltage independent (AMPA)
NMDAsetVm = -30  # Vm NMDA uses when NMDAmode = 0
nmdaTau1 = 2  # rise
nmdaTau2 = 7  # decay
excLock = 0  # NMDA shares onset with E

# voltage dependence of NMDA (used to plug into NMDA mod)
if not alonNMDA:
    NMDA_n = .213  # values from Santhosh
    NMDA_gama = .074
else:
    NMDA_n = .25  # values from Alon
    NMDA_gama = .08
if not NMDAmode:
    NMDA_Voff = 1  # voltage independent
    NMDA_Vset = NMDAsetVm  # g calc from setVm
else:
    NMDA_Voff = 0  # voltage dependent
    NMDA_Vset = 0  # unused (should be anyway -if weird check here)
# -----------------------------------------------------------

# ------------------ SYNAPTIC SETTINGS ---------------------------
quanta = 15  # maximum number of possible events per syn
fewerAMPA = 0  # set AMPA to lower max quanta (shorter release duration)
quantaPrDecr = .95  # % of last quanta Pr
qInterval = 5  # (ms) average quanta interval
qInterVar = 3  # (ms) variance of inter quanta interval

# non-NMDA synaptic settings
inhibTau1 = .5  # inhibitory conductance rise tau (ms)
inhibTau2 = 16  # inhibitory conductance decay tau (ms)
inhibRev = -60  # inhibitory reversal potential (mV)

excTau1 = .1  # excitatory conductance rise tau (ms)
excTau2 = 4  # excitatory conductance decay tau (ms)
excRev = 0  # excitatory reversal potential (mV)

ampaTau1 = .1  # AMPAergic conductance rise tau (ms)
ampaTau2 = 4  # AMPAergic conductance decay tau (ms)

inhibWeight = .004  # weight of inhibitory NetCons (uS)
excWeight = .0005  # weight of excitatory NetCons (uS)
nmdaWeight = .0015
ampaWeight = .0005
# variable weight settings (not in use)
iWeightSD = (inhibWeight/3)/3  # vary between +1/3 and -1/3
eWeightSD = (excWeight/3)/3
variedWeights = 0  # flags if generateWeights has been used
# -----------------------------------------------------------

# ------------ PROBABILITY AND TIMINGS (MANUAL) -------------
# contrast based probability of release
cPi = .8  # inhibition (GABA)
cPe = .5  # excitation (ACH)
cPamp = .5  # excitation (AMPA)
cPn = 0  # excitation (NMDA)
# direction scaling factor (1, dPi*cPi)
dPi = 1  # should probably remove this, never used anyway

succLock = 0  # locks success of E to I (cannot occur without I)

# mean time offset
iOff = -5  # (ms)
eOff = 0  # (ms)

# for flash
meanT = 100  # (ms)
varianceT = 400
# for motion
jitter = 60
# all stim
synVar = 10
nmVar = 7
ampaVar = 7
rho = .8
c = 1  # contrast
# -----------------------------------------------------------

# ------------------VISUAL INPUT ---------------------------
lightstart = 0  # start time of the stimulus bar(ms)
speed = 1  # speed of the stimulus bar (um/ms)
rotateMode = 1  # rotate synapse locations to simulate multiple directions
xMotion = 0  # move bar in x, if not, move bar in y
lightXstart = -60  # start location (X axis)of the stimulus bar (um)
lightXend = 300   # end location (X axis)of the stimulus bar (um)
lightYstart = -70  # start location (Y axis) of the stimulus bar (um)
lightYend = 325   # end location (Y axis) of the stimulus bar (um)
lightReverse = 0  # direction of light sweep (left>right;right<left)
# -----------------------------------------------------------

# ------------------ DIRECTION PARAMETERS -------------------
dirLabel = [225, 270, 315, 0, 45, 90, 135, 180]  # for labelling
inds = np.array(dirLabel).argsort()  # for sorting responses later
circle = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0])  # for polarplot
dirs = [135, 90, 45, 0, 45, 90, 135, 180]  # for reference
nullOnly = 0
if (nullOnly):
    inds = [180]
    dirs = [180]
# parameters
dirTrials = 5
# probabilities
nullPi = cPe + .025
prefPi = cPe/5.0+.025
nonDirectionalPi = 0  # 1 turns off probability scaling of inhibition
# correlations
nullRho = .9  # .8
prefRho = .4
# offsets
nullEoff = -50.0
prefEoff = -50.0
nullIoff = -58
prefIoff = 4

# simple separation of spatial and temporal correlations
diffRhos = 1  # rho is not shared in space and time, use spRho and tmRho
spRho = .9  # space correlation
tmRho = .9  # time correlation
scaleRho = 0  # scale rho with direction
nullTmRho = .9
nullSpRho = .9
prefTmRho = .3
prefSpRho = .3
# -----------------------------------------------------------

# ------------------ OFFSET TESTING PARAMETERS -------------------
# offset
if 0:  # no offsets at all
    nullEoff = 0
    prefEoff = 0
    nullIoff = 0
    prefIoff = 0
if 0:  # symmetrical (non-ds) offsets
    nullEoff = -50
    prefEoff = -50
    nullIoff = -55
    prefIoff = -55
nonDirectionalPi = 1  # non-directional probability of release
fewerAMPA = 0  # fewer AMPA quanta (shorter release duration)
# nullPi, prefPi = 0, 0  # SR
# -----------------------------------------------------------

# ------------------ Dendrite Recordings --------------------
recSyns = 1  # ON/OFF for both modes (list and ALL)
recSynList = [249, 128, 248]  # close triplet
recSynXloc = list(range(len(recSynList)))
recSynYloc = list(range(len(recSynList)))
for i in range(len(recSynList)):
    h.DSGC[0].dend[recSynList[i]].push()
    pts = int(h.n3d())  # number of 3d points of section
    if(pts % 2):  # odd number of points
        recSynXloc[i] = h.x3d((pts-1)/2)
        recSynYloc[i] = h.y3d((pts-1)/2)
    else:
        recSynXloc[i] = (h.x3d(pts/2)+h.x3d((pts/2)-1))/2.0
        recSynYloc[i] = (h.y3d(pts/2)+h.y3d((pts/2)-1))/2.0
    h.pop_section()
# or just record ALL of them
recAllSyns = 0
# OR record the whole tree
recWholeTree = 1
recSegXloc = list(range(len(allDends)*dendSegs*2))
recSegYloc = list(range(len(allDends)*dendSegs*2))
seg = 0
for i in range(len(allDends)):
    allDends[i].push()
    pts = int(h.n3d())  # number of 3d points of section
    for s in range(dendSegs*2):
        if(pts % 2):  # odd number of points
            recSegXloc[seg] = h.x3d(s*(pts-1)/(2*dendSegs))
            recSegYloc[seg] = h.y3d(s*(pts-1)/(2*dendSegs))
        else:
            recSegXloc[seg] = (h.x3d(s*(pts-1)/(2*dendSegs)) +
                               h.x3d(s*(pts-1)/(2*dendSegs))-1)/2.0
            recSegYloc[seg] = (h.y3d(s*(pts-1)/(2*dendSegs)) +
                               h.y3d(s*(pts-1)/(2*dendSegs))-1)/2.0
        seg += 1
    h.pop_section()

# factor by which number of samples of stored recordings will be reduced
downsample = .1  # 1 will leave samplerate as is (= timesteps)
# ------------------------------------------------------------

seed = 0  # 10000#1
nzSeed = 0  # 10000
h('progress = 0.0')

# sort dendrite sections by branch order and if they are terminal
soma.push()  # start from soma
orderPos = [0]  # last item indicates which # child to check next
orderList = [[]]  # list of lists, dends sorted by branch order
terminals = []  # terminal branches (no children)
nonTerms = []  # non-terminal branches (with children)
keepGoing = 1
while keepGoing:
    sref = h.SectionRef()  # reference to current section
    if orderPos[-1] < sref.nchild():  # more children to check
        if len(orderPos) > 1:  # exclude primes from non-terminals
            nonTerms.append(h.cas())  # add this parent dend to list
        sref.child[orderPos[-1]].push()  # access child of current
        if len(orderPos) > len(orderList):
            orderList.append([])
        orderList[len(orderPos)-1].append(h.cas())  # order child
        orderPos.append(0)  # extend to next order
    else:  # ran out of children for current parent
        if len(orderPos) == 1:  # exceeded number of prime dends
            keepGoing = 0  # entire tree is mapped, exit loop
        else:  # current part of tree is exhausted, walk back an order
            if not sref.nchild():  # no children at all
                terminals.append(h.cas())  # add childless dend to list
            del orderPos[-1]  # step back an order
            orderPos[-1] += 1  # go to next child of previous order
            h.pop_section()  # go back to parent
print("Number of terminal dendrites: " + str(len(terminals)))


def distCalc():
    '''
    Get the location of each synapse or recording location and the cable
    distances (running along dendrites) between each of them.
    '''
    # calculate branching distance between recording locations
    termMode = 0  # if 1, do terminals only, if 0 do all segs
    if termMode:
        sectionList = terminals
        numLocs = len(terminals)
    else:
        sectionList = allDends
        numLocs = len(allDends)*2*dendSegs

    locs = {'X': [], 'Y': []}
    dendNums = []
    for i in range(len(sectionList)):
        sectionList[i].push()
        pts = int(h.n3d())  # number of 3d points of section
        for s in range(dendSegs*2):
            if(pts % 2):  # odd number of points
                locs['X'].append(h.x3d(s*(pts-1)/(2*dendSegs)))
                locs['Y'].append(h.y3d(s*(pts-1)/(2*dendSegs)))
            else:
                locs['X'].append((h.x3d(s*(pts-1)/(2*dendSegs))
                                 + h.x3d(s*(pts-1)/(2*dendSegs))-1)/2.0)
                locs['Y'].append((h.y3d(s*(pts-1)/(2*dendSegs))
                                 + h.y3d(s*(pts-1)/(2*dendSegs))-1)/2.0)

        name = h.secname()
        dendNums.append(name.replace('DSGC[0].dend[', '').replace(']', ''))
        h.pop_section()

    # print coordinates of all terminal dendrites to file
    if termMode:
        locfname = 'terminalLocations.csv'
        dendfname = 'termDendNumbers.csv'
    else:
        locfname = 'recLocations.csv'
        dendfname = 'dendNumbers.csv'
    # x, y coordinates for each recording location
    locations = pd.DataFrame(locs)
    locations.to_csv(basest+locfname, index=False)
    # python index and neuron dend numbers
    dendNumbers = pd.DataFrame(dendNums, columns=['dendNum'])
    dendNumbers.to_csv(basest+dendfname, index_label='pyIdx')

    distBetwRecs = list(range(numLocs))
    for i in range(numLocs):
        distBetwRecs[i] = np.zeros(numLocs)
    iCnt = 0
    for i in range(len(sectionList)):
        sectionList[i].push()
        for iSeg in range(dendSegs*2):
            h.distance(0, iSeg*segStep)  # set origin as middle of current sec
            kCnt = 0
            for k in range(len(allDends)):
                sectionList[k].push()
                for kSeg in range(dendSegs*2):
                    distBetwRecs[iCnt][kCnt] = h.distance(kSeg*segStep)
                    kCnt += 1
                h.pop_section()
            iCnt += 1
        h.pop_section()

    dists = pd.DataFrame(distBetwRecs)
    dists.to_csv(basest+'distBetwRecs.csv', header=None, index=False)


def setSoma():
    '''Set membrane properties of soma compartment'''
    soma.nseg = 1
    soma.Ra = 100

    if activeSOMA:
        soma.insert('HHst')
        soma.gnabar_HHst = somaNa
        soma.gkbar_HHst = somaK
        soma.gkmbar_HHst = .003  # replace with var
        soma.gleak_HHst = soma_gleak_hh  # (S/cm2)
        soma.eleak_HHst = soma_eleak_hh
        soma.NF_HHst = soma_nzFactor
    else:
        soma.insert('pas')
        soma.g_pas = soma_gleak_pas  # (S/cm2)
        soma.e_pas = soma_eleak_pas


setSoma()


def membSetup():
    '''Set membrane properties of dendrite compartments'''
    # activeSYN for only making branches that have synapses active
    if activeSYN:
        for dend in terminals:
            dend.insert('HHst')
            dend.gnabar_HHst = dendNa
            dend.gkbar_HHst = dendK
            dend.gkmbar_HHst = .0004  # replace with var
            dend.gleak_HHst = dend_gleak_hh
            dend.eleak_HHst = dend_eleak_hh
            dend.NF_HHst = dend_nzFactor
        if dendPas:
            for dend in nonTerms:
                dend.insert('pas')
                dend.g_pas = dend_gleak_pas
                dend.e_pas = dend_eleak_pas
        else:
            for dend in nonTerms:
                dend.insert('HHst')
                dend.gnabar_HHst = 0
                dend.gkbar_HHst = dendK
                dend.gkmbar_HHst = .0008  # replace with var
                dend.gleak_HHst = dend_gleak_hh
                dend.eleak_HHst = dend_eleak_hh
                dend.NF_HHst = dend_nzFactor
    else:
        for order in orderList[1:]:  # except primes
            if activeDEND:
                for dend in order:
                    dend.insert('HHst')
                    dend.gnabar_HHst = dendNa
                    dend.gkbar_HHst = dendK
                    dend.gkmbar_HHst = .0008  # replace with var
                    dend.gleak_HHst = dend_gleak_hh
                    dend.eleak_HHst = dend_eleak_hh
                    dend.NF_HHst = dend_nzFactor
            elif dendPas:
                for dend in order:
                    dend.insert('pas')
                    dend.g_pas = dend_gleak_pas
                    dend.e_pas = dend_eleak_pas
            else:
                for dend in order:
                    dend.insert('HHst')
                    dend.gnabar_HHst = 0
                    dend.gkbar_HHst = dendK
                    dend.gkmbar_HHst = .0008  # replace with var
                    dend.gleak_HHst = dend_gleak_hh
                    dend.eleak_HHst = dend_eleak_hh
                    dend.NF_HHst = dend_nzFactor

    # prime dendrites
    for dend in orderList[0]:
        if activeDEND:
            dend.insert('HHst')
            dend.gnabar_HHst = primNa
            dend.gkbar_HHst = primK
            dend.gleak_HHst = prim_gleak_hh
            dend.eleak_HHst = prim_eleak_hh
            dend.NF_HHst = dend_nzFactor
        else:
            dend.insert('pas')
            dend.g_pas = prim_gleak_pas
            dend.e_pas = prim_eleak_pas

    # all dendrites
    for order in orderList:
        for dend in order:
            dend.nseg = dendSegs
            dend.Ra = 100
            if activeDEND:
                dend.gtbar_HHst = .0003  # default
                dend.glbar_HHst = .0003  # default


membSetup()


def setSyn():
    '''
    Create synapses (Syn, Stim and Conn NEURON objects working together).
    E is Ach, I is GABA. (X, Y) coordinates are used to determine activation
    timing by simulated light bar stimulus.
    '''
    global Esyn, Isyn, Estim, Istim, Econ, Icon, AMPsyn, AMPstim, AMPcon
    global NMsyn, NMstim, NMcon, xLocs, yLocs

    # number of synapses (just on terminal branches now)
    h('nSyn = 0')
    if termSynOnly:
        h.nSyn = len(terminals)
    else:
        for order in orderList[1:]:
            for dend in order:
                h.nSyn += 1
    nSyn = int(h.nSyn)
    # create hoc objrefs for all synapses (hocs objects for GUI)
    h('objref Esyn[nSyn], Isyn[nSyn], AMPsyn[nSyn], NMsyn[nSyn]')
    Esyn, Isyn, AMPsyn, NMsyn = h.Esyn, h.Isyn, h.AMPsyn, h.NMsyn

    Estim, Istim, NMstim = [list(range(nSyn)) for i in range(3)]
    Econ, Icon, NMcon = [list(range(nSyn)) for i in range(3)]
    AMPstim, AMPcon = [list(range(nSyn)) for i in range(2)]

    xLocs = list(range(int(h.nSyn)))
    yLocs = list(range(int(h.nSyn)))

    syndex = 0  # synapse index corresponds to place in terminal list
    if termSynOnly:
        for i in range(len(terminals)):
            terminals[i].push()

            # 3D location of the synapses on this dendrite
            # place them in the middle since only one syn per dend
            pts = int(h.n3d())
            if(pts % 2):  # odd number of points
                xLocs[i] = h.x3d((pts-1)/2)
                yLocs[i] = h.y3d((pts-1)/2)
            else:
                xLocs[i] = (h.x3d(pts/2)+h.x3d((pts/2)-1))/2.0
                yLocs[i] = (h.y3d(pts/2)+h.y3d((pts/2)-1))/2.0

            Isyn[i] = h.Exp2Syn(.5)
            Isyn[i].tau1 = inhibTau1  # rise
            Isyn[i].tau2 = inhibTau2  # decay

            Esyn[i] = h.Exp2Syn(.5)
            Esyn[i].tau1 = excTau1  # rise
            Esyn[i].tau2 = excTau2  # decay

            AMPsyn[i] = h.Exp2Syn(.5)
            AMPsyn[i].tau1 = ampaTau1  # rise
            AMPsyn[i].tau2 = ampaTau2  # decay

            NMsyn[i] = h.Exp2NMDA(.5)
            NMsyn[i].tau1 = nmdaTau1  # rise
            NMsyn[i].tau2 = nmdaTau2  # decay

            Isyn[i].e = inhibRev  # reversal potential
            Esyn[i].e = excRev
            AMPsyn[i].e = excRev
            NMsyn[i].e = excRev

            # create NetStims to drive the synapses through NetCons
            Istim[i] = []
            Estim[i] = []
            AMPstim[i] = []
            NMstim[i] = []
            for q in range(quanta):
                Istim[i].append(h.NetStim(.5))
                Istim[i][q].interval = 0
                Istim[i][q].number = 1
                Istim[i][q].noise = 0

                Estim[i].append(h.NetStim(.5))
                Estim[i][q].interval = 0
                Estim[i][q].number = 1
                Estim[i][q].noise = 0

                AMPstim[i].append(h.NetStim(.5))
                AMPstim[i][q].interval = 0
                AMPstim[i][q].number = 1
                AMPstim[i][q].noise = 0

                NMstim[i].append(h.NetStim(.5))
                NMstim[i][q].interval = 0
                NMstim[i][q].number = 1
                NMstim[i][q].noise = 0

            # NMDA voltage settings
            NMsyn[i].n = NMDA_n
            NMsyn[i].gama = NMDA_gama
            NMsyn[i].Voff = NMDA_Voff
            NMsyn[i].Vset = NMDA_Vset

            # create the NetCons (link NetStims and Syns)
            Icon[i] = []
            Econ[i] = []
            AMPcon[i] = []
            NMcon[i] = []
            for q in range(quanta):
                # change to 0 delay in future, data used 10
                Icon[i].append(h.NetCon(
                    Istim[i][q], Isyn[i], 0, 10, inhibWeight))
                Econ[i].append(h.NetCon(
                    Estim[i][q], Esyn[i], 0, 10, excWeight))
                AMPcon[i].append(h.NetCon(
                    AMPstim[i][q], AMPsyn[i], 0, 10, ampaWeight))
                NMcon[i].append(h.NetCon(
                    NMstim[i][q], NMsyn[i], 0, 10, nmdaWeight))

            h.pop_section()
    else:
        for order in orderList[1:]:
            for dend in order:
                dend.push()
                # 3D location of the synapses on this dendrite
                # place them in the middle since only one set per dend
                pts = int(h.n3d())
                if(pts % 2):  # odd number of points
                    xLocs[syndex] = h.x3d((pts-1)/2)
                    yLocs[syndex] = h.y3d((pts-1)/2)
                else:
                    xLocs[syndex] = (h.x3d(pts/2)+h.x3d((pts/2)-1))/2.0
                    yLocs[syndex] = (h.y3d(pts/2)+h.y3d((pts/2)-1))/2.0

                Isyn[syndex] = h.Exp2Syn(.5)
                Isyn[syndex].tau1 = inhibTau1  # rise
                Isyn[syndex].tau2 = inhibTau2  # decay

                Esyn[syndex] = h.Exp2Syn(.5)
                Esyn[syndex].tau1 = excTau1  # rise
                Esyn[syndex].tau2 = excTau2  # decay

                AMPsyn[syndex] = h.Exp2Syn(.5)
                AMPsyn[syndex].tau1 = ampaTau1  # rise
                AMPsyn[syndex].tau2 = ampaTau2  # decay

                NMsyn[syndex] = h.Exp2NMDA(.5)
                NMsyn[syndex].tau1 = nmdaTau1  # rise
                NMsyn[syndex].tau2 = nmdaTau2  # decay

                Isyn[syndex].e = inhibRev  # reversal potential
                Esyn[syndex].e = excRev
                AMPsyn[syndex].e = excRev
                NMsyn[syndex].e = excRev

                # create NetStims to drive the synapses through NetCons
                # each syn will have as many stims as max quanta
                Istim[syndex] = []
                Estim[syndex] = []
                AMPstim[syndex] = []
                NMstim[syndex] = []
                for i in range(quanta):
                    Istim[syndex].append(h.NetStim(.5))
                    Istim[syndex][i].interval = 0
                    Istim[syndex][i].number = 1
                    Istim[syndex][i].noise = 0

                    Estim[syndex].append(h.NetStim(.5))
                    Estim[syndex][i].interval = 0
                    Estim[syndex][i].number = 1
                    Estim[syndex][i].noise = 0

                    AMPstim[syndex].append(h.NetStim(.5))
                    AMPstim[syndex][i].interval = 0
                    AMPstim[syndex][i].number = 1
                    AMPstim[syndex][i].noise = 0

                    NMstim[syndex].append(h.NetStim(.5))
                    NMstim[syndex][i].interval = 0
                    NMstim[syndex][i].number = 1
                    NMstim[syndex][i].noise = 0

                # NMDA voltage settings
                NMsyn[syndex].n = NMDA_n
                NMsyn[syndex].gama = NMDA_gama
                NMsyn[syndex].Voff = NMDA_Voff
                NMsyn[syndex].Vset = NMDA_Vset

                # create the NetCons (link NetStims and Syns)
                Icon[syndex] = []
                Econ[syndex] = []
                AMPcon[syndex] = []
                NMcon[syndex] = []
                for i in range(quanta):
                    # change to 0 delay in future, data used 10
                    Icon[syndex].append(h.NetCon(
                        Istim[syndex][i], Isyn[syndex], 0, 10, inhibWeight))
                    Econ[syndex].append(h.NetCon(
                        Estim[syndex][i], Esyn[syndex], 0, 10, excWeight))
                    AMPcon[syndex].append(h.NetCon(
                        AMPstim[syndex][i], AMPsyn[syndex], 0, 10, ampaWeight))
                    NMcon[syndex].append(h.NetCon(
                        NMstim[syndex][i], NMsyn[syndex], 0, 10, nmdaWeight))

                h.pop_section()
                syndex += 1


setSyn()


def flashOnsets(seedL):
    '''
    Calculate onset times for each synapse randomly as though stimulus was a
    flash. Timing jitter is applied using pseudo-random number generators.
    '''
    global seed, iTimes, eTimes, nTimes, ampTimes

    if diffRhos:
        rho = tmRho  # this rho is local

    meanOnset = list(range(int(h.nSyn)))
    mOn = h.Random(seedL)
    mOn.normal(meanT, varianceT)
    seedL += 1

    iTimes, eTimes, nTimes, ampTimes = [[] for i in range(4)]

    meanOnset = [mOn.repick() for i in range(len(meanOnset))]

    for syn in range(int(h.nSyn)):
        for q in range(quanta):
            iOn = h.Random(seedL)
            iOn.normal(0, 1)
            seedL += 1
            iOnPick = iOn.repick()

            nOn = h.Random(seedL)
            nOn.normal(0, 1)
            seedL += 1
            nOnPick = nOn.repick()

            ampOn = h.Random(seedL)
            ampOn.normal(0, 1)
            seedL += 1
            ampOnPick = nOn.repick()

            if not q:
                onset = meanOnset[syn]
            else:
                # add variable delay til next quanta (if there is one)
                quantDelay = h.Random(seedL)
                quantDelay.normal(qInterval, qInterVar)  # test 10ms avg delay
                seedL += 1
                onset += quantDelay.repick()  # update avg onset time

            Istim[syn][q].start = iOnPick*synVar+onset+iOff

            eRand = h.Random(seedL)
            eRand.normal(0, 1)
            seedL += 1
            eRandPick = eRand.repick()

            # y1picks = rho*x1picks + sqrt(1-rho^2)*x2picks
            eOnPick = np.multiply(iOnPick, rho)
            temp = np.multiply(eRandPick, np.sqrt(1.0-rho**2))
            eOnPick = np.add(eOnPick, temp)
            Estim[syn][q].start = eOnPick*synVar+onset+eOff

            # set NMDA onset
            if excLock:
                NMstim[syn][q].start = Estim[syn][q].start
            else:
                NMstim[syn][q].start = nOnPick*nmVar+onset
            # set AMPA onset
            AMPstim[syn][q].start = ampOnPick*ampaVar+onset

            iTimes.append(Istim[syn][q].start - iOff - onset)
            eTimes.append(Estim[syn][q].start - eOff - onset)
            ampTimes.append(AMPstim[syn][q].start - onset)
            nTimes.append(NMstim[syn][q].start - onset)

    # print(st.pearsonr(eTimes, iTimes))
    seed = seedL


def barOnsets(seedL, _xLocs, _yLocs):
    '''
    Calculate onset times for each synapse based on when the simulated bar
    would be passing over their location, modified by spatial offsets. Timing
    jitter is applied using pseudo-random number generators.
    '''
    global seed, iTimes, eTimes, ampTimes, nTimes,  synTimes

    if diffRhos:
        rho = tmRho  # this rho is local

    # distributions of all onsets (0 centred)
    synTimes, iTimes, eTimes, ampTimes, nTimes = [], [], [], [], []

    for syn in range(int(h.nSyn)):
        # distance to synapse divided by speed
        if xMotion:
            if lightReverse:
                synT = (lightstart+(lightXend-_xLocs[syn])/speed)
            else:
                synT = (lightstart+(_xLocs[syn]-lightXstart)/speed)
        else:  # motion in y
            if lightReverse:
                synT = (lightstart+(lightYend-_yLocs[syn])/speed)
            else:
                synT = (lightstart+(_yLocs[syn]-lightYstart)/speed)

        for i in range(quanta):
            # mean onset time for current synapse
            if not i:
                mOn = h.Random(seedL)
                mOn.normal(synT, jitter)
                seedL += 1
                synOnset = mOn.repick()
                synTimes.append(synOnset)
            else:
                # add variable delay til next quanta (if there is one)
                quantDelay = h.Random(seedL)
                quantDelay.normal(qInterval, qInterVar)  # test 10ms avg delay
                seedL += 1
                synOnset += quantDelay.repick()  # update avg synapse time

            # inhib
            iOn = h.Random(seedL)
            iOn.normal(0, 1)
            seedL += 1
            iOnPick = iOn.repick()
            # nmda
            nOn = h.Random(seedL)
            nOn.normal(0, 1)
            seedL += 1
            nOnPick = nOn.repick()
            # ampa
            ampOn = h.Random(seedL)
            ampOn.normal(0, 1)
            seedL += 1
            ampOnPick = ampOn.repick()
            # ach
            eRand = h.Random(seedL)
            eRand.normal(0, 1)
            seedL += 1
            eRandPick = eRand.repick()

            Istim[syn][i].start = iOnPick*synVar+synOnset+(iOff/speed)

            # y1picks = rho*x1picks + sqrt(1-rho^2)*x2picks
            eOnPick = np.multiply(iOnPick, rho)
            temp = np.multiply(eRandPick, np.sqrt(1.0-rho**2))
            eOnPick = np.add(eOnPick, temp)
            Estim[syn][i].start = eOnPick*synVar+synOnset+(eOff/speed)

            # set NMDA onset
            if excLock:
                NMstim[syn][i].start = Estim[syn][i].start
            else:
                NMstim[syn][i].start = nOnPick*nmVar+synOnset
            # set AMPA onset
            AMPstim[syn][i].start = ampOnPick*ampaVar+synOnset

            iTimes.append(Istim[syn][i].start)  # - synOnset - iOff
            eTimes.append(Estim[syn][i].start)  # - synOnset - eOff
            ampTimes.append(AMPstim[syn][i].start)
            nTimes.append(NMstim[syn][i].start)  # - synOnset
    seed = seedL


def setFailures(seedL):
    '''
    Determine number of quantal activations of each synapse occur on a trial.
    Psuedo-random numbers generated for each synapse are compared against
    thresholds set by probability of release to determine if the "pre-synapse"
    succeeds or fails to release neurotransmitter.
    '''
    global seed, iSucc, eSucc, nSucc, iPicks, ePicks, nPicks

    if diffRhos:  # change local rho to spatial specific
        rho = spRho
    rho = .986 if rho > .986 else rho  # numbers above can result in NaNs

    # calculate input rho required to achieve the desired output rho
    # exponential fit: y = y0 + A * exp(-invTau * x)
    # y0 = 1.0461; A = -0.93514; invTau = 3.0506
    rho = 1.0461 - 0.93514 * np.exp(-3.0506 * rho)

    iRand = h.Random(seedL)
    iRand.normal(0, 1)
    seedL += 1

    eRand = h.Random(seedL)
    eRand.normal(0, 1)
    seedL += 1

    ampRand = h.Random(seedL)
    ampRand.normal(0, 1)
    seedL += 1

    nRand = h.Random(seedL)
    nRand.normal(0, 1)
    seedL += 1

    iSucc, eSucc, ampSucc, nSucc = [], [], [], []

    iPicks = [iRand.repick() for i in range(int(h.nSyn))]
    ePicks = [eRand.repick() for i in range(int(h.nSyn))]
    ampPicks = [ampRand.repick() for i in range(int(h.nSyn))]
    nPicks = [nRand.repick() for i in range(int(h.nSyn))]
    # correlate ACH with GABA
    tempVec1 = np.multiply(iPicks, rho)
    tempVec2 = np.multiply(ePicks, np.sqrt(1.0-rho**2))
    ePicks = np.add(tempVec1, tempVec2)

    # now from pick distributions, determine success/failures
    for i in range(int(h.nSyn)):
        qPr = 1.0  # decreases with each possible quanta
        for q in range(quanta):
            # inhibition
            if (st.norm.ppf((1-cPi*dPi*qPr)/2.0)*np.std(iPicks) < iPicks[i]
                    < st.norm.ppf(1-(1-cPi*dPi*qPr)/2.0)*np.std(iPicks)):
                iSucc.append(1)
                Istim[i][q].number = 1
            else:
                iSucc.append(0)
                Istim[i][q].number = 0
            # ach
            if succLock and iSucc[q]:  # E success tied to I
                if (st.norm.ppf((1-cPe*qPr)/2.0)*np.std(ePicks) < ePicks[i]
                        < st.norm.ppf(1-(1-cPe*qPr)/2.0)*np.std(ePicks)):
                    eSucc.append(1)
                    Estim[i][q].number = 1
                else:
                    eSucc.append(0)
                    Estim[i][q].number = 0
            elif succLock and not iSucc[i]:
                eSucc.append(0)
                Estim[i][q].number = 0
            elif (st.norm.ppf((1-cPe*qPr)/2.0)*np.std(ePicks) < ePicks[i]
                    < st.norm.ppf(1-(1-cPe*qPr)/2.0)*np.std(ePicks)):
                eSucc.append(1)
                Estim[i][q].number = 1
            else:
                eSucc.append(0)
                Estim[i][q].number = 0
            # nmda
            if (st.norm.ppf((1-cPn*qPr)/2.0)*np.std(nPicks) < nPicks[i]
                    < st.norm.ppf(1-(1-cPn*qPr)/2.0)*np.std(nPicks)):
                nSucc.append(1)
                NMstim[i][q].number = 1
            else:
                nSucc.append(0)
                NMstim[i][q].number = 0
            # ampa
            if not fewerAMPA or q < quanta/2:
                if (st.norm.ppf((1-cPamp*qPr)/2.0)*np.std(ampPicks)
                        < ampPicks[i] <
                        st.norm.ppf(1-(1-cPamp*qPr)/2.0)*np.std(ampPicks)):
                    ampSucc.append(1)
                    AMPstim[i][q].number = 1
                else:
                    ampSucc.append(0)
                    AMPstim[i][q].number = 0
            else:
                ampSucc.append(0)
                AMPstim[i][q].number = 0

            qPr = qPr * quantaPrDecr  # Pr of next possible quanta decreases
    # print(st.pearsonr(eSucc, iSucc))
    pearson = st.pearsonr(eSucc, iSucc)
    seed = seedL
    return pearson[0]


def dataRun(_xLocs=xLocs, _yLocs=yLocs, vcMode=0, dirRunning=0):
    '''
    Initialize model, get synapse onset and release numbers, update membrane
    noise seeds and run the model. Calculate somatic response and return to
    calling function.
    '''
    global nzSeed

    threshCount = 0
    h.progress = 0.0 if not dirRunning else h.progress

    h.init()
    # flashOnsets(seed)
    barOnsets(seed, _xLocs, _yLocs)
    setFailures(seed)

    # set HHst noise seeds
    if activeSOMA:
        soma.seed_HHst = nzSeed
        nzSeed += 1
    if activeSYN:
        for dend in terminals:
            dend.seed_HHst = nzSeed
            nzSeed += 1
    elif activeDEND:
        for order in orderList[1:]:  # except primes
            for dend in order:
                dend.seed_HHst = nzSeed
                nzSeed += 1
    # prime dendrites
    if activeDEND:  # regardless if activeSYN
        for dend in orderList[0]:
            dend.seed_HHst = nzSeed
            nzSeed += 1

    vecVm = h.Vector()
    vecVm.record(soma(.5)._ref_v)

    h.run()

    # change threshold to spikes if active conductances on
    if not vcMode:
        if TTX:
            Vm = np.array(vecVm)
            psp = Vm + 61.3
            # store area of PSP
            threshCount += sum(psp[70:])/len(psp[70:])
            spkTs = []  # empty
        else:
            spkCount, spkTs = findSpikes(np.array(vecVm), thresh=20)
            threshCount += spkCount
    else:
        threshCount, spkTs, vecVm = [], [], []

    return threshCount, spkTs, vecVm


def dirRun():
    '''
    Run model through 8 directions for a number of trials and save the data.
    Offets and probabilities of release for inhibition are updated here
    before calling dataRun() to execute the model.
    '''
    global iOff, cPi, dirRunning, rho, tmRho, spRho, rhoMetrics, eOff
    tempRho = rho
    h.progress = 0.0

    dirIoff, dirEoff, dirPi = [], [], []
    dirRho, dirTmRho, dirSpRho = [], [], []
    for i in range(len(dirs)):
        # sigmoidal scaling of offsets, amplitudes, etc over direction
        dirIoff.append(prefIoff-(prefIoff-nullIoff)*(
            1.0 - .98/(1.0 + np.exp((dirs[i] - 74.69)/24.36))))
        dirEoff.append(prefEoff-(prefEoff-nullEoff)*(
            1.0 - .98/(1.0 + np.exp((dirs[i] - 91.0)/25.0))))
        dirPi.append(prefPi+(nullPi-prefPi)*(
            1.0 - .98/(1.0 + np.exp((dirs[i] - 91.0)/25.0))))
        dirRho.append(prefRho+(nullRho-prefRho)*(
            1.0 - .98/(1.0 + np.exp((dirs[i] - 91.0)/25.0))))
        dirTmRho.append(prefTmRho+(nullTmRho-prefTmRho)*(
            1.0 - .98/(1.0 + np.exp((dirs[i] - 91.0)/25.0))))
        dirSpRho.append(prefSpRho+(nullSpRho-prefSpRho)*(
            1.0 - .98/(1.0 + np.exp((dirs[i] - 91.0)/25.0))))

    dirSpks, dirSpkTimes = [list(range(len(dirs))) for i in range(2)]
    trialSpks, DSi, theta = [list(range(dirTrials)) for i in range(3)]

    VmRecs, iCaRecs = [list(range(dirTrials)) for i in range(2)]
    dirSynRecs, iCaDirSynRecs = [list(range(len(dirs))) for i in range(2)]
    trialSynRecs, iCaTrialSynRecs = [list(range(dirTrials)) for i in range(2)]

    if recWholeTree:
        nRecs = len(allDends)*2*dendSegs
    elif recAllSyns:
        nRecs = len(terminals)
    else:
        nRecs = len(recSynList)

    synRecs, iCaSynRecs = [list(range(nRecs)) for i in range(2)]

    if rotateMode:
        origin = findOrigin(allDends)

    for j in range(dirTrials):
        VmRecs[j], iCaRecs[j] = [list(range(len(dirs))) for i in range(2)]
        for i in range(len(dirs)):
            eOff = dirEoff[i]
            iOff = dirIoff[i]
            if not nonDirectionalPi:
                cPi = dirPi[i]
            else:
                cPi = nullPi

            if rotateMode:
                dirXlocs, dirYlocs = rotate(
                    origin, xLocs, yLocs, np.radians(dirLabel[i]))
            else:
                dirXlocs, dirYlocs = xLocs, yLocs

            if scaleRho:
                # rho = dirRho[i]
                spRho = dirSpRho[i]
                tmRho = dirTmRho[i]
            if TTX and recSyns:
                if recWholeTree:
                    seg = 0
                    for k in range(len(allDends)):
                        for s in range(2*dendSegs):
                            synRecs[seg] = h.Vector()  # Vm
                            synRecs[seg].record(allDends[k](s*segStep)._ref_v)

                            iCaSynRecs[seg] = h.Vector()  # iCa
                            iCaSynRecs[seg].record(
                                allDends[k](s*segStep)._ref_ica)

                            seg += 1  # next seg
                elif recAllSyns:
                    for k in range(len(synRecs)):
                        synRecs[k] = h.Vector()  # Vm
                        synRecs[k].record(terminals[k](.5)._ref_v)

                        iCaSynRecs[k] = h.Vector()  # iCa
                        iCaSynRecs[k].record(terminals[k](.5)._ref_ica)
                else:
                    for k in range(len(synRecs)):
                        synRecs[k] = h.Vector()  # Vm
                        synRecs[k].record(
                            h.DSGC[0].dend[recSynList[k]](.5)._ref_v)

                        iCaSynRecs[k] = h.Vector()  # iCa
                        iCaSynRecs[k].record(
                            h.DSGC[0].dend[recSynList[k]](.5)._ref_ica)

            threshCount, spkTs, vecVm = dataRun(
                _xLocs=dirXlocs, _yLocs=dirYlocs, dirRunning=1)

            # store vectors in list between runs
            VmRecs[j][i] = np.round(np.array(vecVm), decimals=3)
            dirSpks[i] = threshCount  # spike number
            dirSpkTimes[i] = spkTs[:]  # spike times

            if TTX and recSyns:
                for k in range(len(synRecs)):
                    # Vm
                    synRecs[k] = np.round(
                        synRecs[k].resample(synRecs[k], downsample),
                        decimals=3)
                    # iCa
                    iCaSynRecs[k] = np.round(
                        iCaSynRecs[k].resample(iCaSynRecs[k], downsample),
                        decimals=6)

                dirSynRecs[i] = synRecs[:]  # Vm
                iCaDirSynRecs[i] = iCaSynRecs[:]  # iCa

            xpts = np.multiply(dirSpks, np.cos(np.radians(dirLabel)))
            ypts = np.multiply(dirSpks, np.sin(np.radians(dirLabel)))
            xsum = sum(xpts)
            ysum = sum(ypts)
            DSi[j] = np.sqrt(xsum**2 + ysum**2)/sum(dirSpks)
            theta[j] = np.arctan2(ysum, xsum)*180/np.pi
            trialSpks[j] = dirSpks[:]
            if TTX and recSyns:
                trialSynRecs[j] = dirSynRecs[:]  # Vm
                iCaTrialSynRecs[j] = iCaDirSynRecs[:]  # iCa

        h.progress = h.progress + 100.0/dirTrials

    # total spikes in each direction and avg DSi and theta
    dirSpks = np.zeros(8)
    for trial in trialSpks:
        dirSpks += trial
        xpts = np.multiply(dirSpks, np.cos(np.radians(dirLabel)))
        ypts = np.multiply(dirSpks, np.sin(np.radians(dirLabel)))
        xsum = sum(xpts)
        ysum = sum(ypts)
        avgDSi = np.sqrt(xsum**2 + ysum**2)/sum(dirSpks)
        avgtheta = np.arctan2(ysum, xsum)*180/np.pi

    print('spRho: ' + str(spRho) + '\t tmRho: ' + str(tmRho))
    print('dirIoff:')
    print(np.round(dirIoff, decimals=3))
    print('dirPi:')
    print(np.round(dirPi, decimals=3))
    # print('dirRho:')
    # print(np.round(dirRho,decimals=3))
    print('total spikes:')
    print(dirSpks)
    print('avg DSi: ' + str(np.round(avgDSi, decimals=3)))
    print('avg theta: ' + str(np.round(avgtheta, decimals=2)))
    print('DSis:')
    print(np.round(DSi, decimals=3))
    print('	sdev: ' + str(np.round(np.std(DSi), decimals=2)))
    print('thetas:')
    print(np.round(theta, decimals=2))
    print('	sdev: ' + str(np.round(np.std(theta), decimals=2)))

    # Hierarchical Data Format file for spiking data
    dirSpksDF = pd.DataFrame(np.array(trialSpks).T, index=dirLabel)
    dirSpksDF.to_hdf(basest+runLabel+'spikeData.h5', key='dirSpks', mode='w')
    dirInputsDF = pd.DataFrame({'dirPi': dirPi, 'dirIoff': dirIoff},
                               index=dirLabel)
    dirInputsDF.to_hdf(basest+runLabel+'spikeData.h5', key='dirInputs',
                       mode='a')
    # save the last seeds used for inputs and noise
    lastSeedsDF = pd.DataFrame({'seed': seed-1, 'nzSeed': nzSeed-1}, index=[0])
    lastSeedsDF.to_csv(basest+runLabel+'lastSeeds.csv', index=False)

    # rehape to 2D array (time, dirs*trials)
    VmRecs = np.array(VmRecs).T.reshape(len(vecVm), -1)
    # Hierarchical columns for trial and direction
    mi = pd.MultiIndex.from_product(
        [np.arange(dirTrials), dirLabel], names=('trials', 'direction'))
    VmDF = pd.DataFrame(VmRecs, columns=mi)
    VmDF.to_hdf(basest+runLabel+'spikeData.h5', key='Vm', mode='a')

    if TTX and recSyns:
        vmTreeRecs = np.array(trialSynRecs).T.reshape(len(synRecs[0]), -1)
        mi = pd.MultiIndex.from_product(
            [np.arange(dirTrials), dirLabel, np.arange(len(synRecs))],
            names=('trials', 'direction', 'synapse'))
        vmTreeDF = pd.DataFrame(vmTreeRecs, columns=mi)
        vmTreeDF.to_hdf(basest+runLabel+'treeRecData.h5', key='Vm', mode='w')

        iCaTreeRecs = np.array(trialSynRecs).T.reshape(len(synRecs[0]), -1)
        mi = pd.MultiIndex.from_product(
            [np.arange(dirTrials), dirLabel, np.arange(len(synRecs))],
            names=('trials', 'direction', 'synapse'))
        iCaTreeDF = pd.DataFrame(iCaTreeRecs, columns=mi)
        iCaTreeDF.to_hdf(basest+runLabel+'treeRecData.h5', key='iCa', mode='a')

        locations = pd.DataFrame({'X': recSegXloc, 'Y': recSegYloc})
        locations.to_csv(basest+runLabel+'treeRecLocations.csv', index=False)

    # make polar plot with all trials
    theta = np.deg2rad(theta)
    polar = plt.subplot(111, projection='polar')
    peakSpk = 0
    for i in range(dirTrials):
        if np.max(trialSpks[i]) > peakSpk:
            peakSpk = np.max(trialSpks[i])
            circSpks = np.array(trialSpks[i])
            circSpks = circSpks[inds]
            circSpks = np.append(circSpks, circSpks[0])
            polar.plot(circle, circSpks, '.75')
    for i in range(dirTrials):
        # DSi is on a scale of 0 to 1
        polar.plot([theta[i], theta[i]], [0.0, DSi[i]*peakSpk], '.75')
    # add in avg values
    circSpks = np.array(dirSpks)/float(dirTrials)
    circSpks = circSpks[inds]  # organize directions into circle
    circSpks = np.append(circSpks, circSpks[0])
    avgtheta = np.deg2rad(avgtheta)
    polar.plot(circle, circSpks, '0.')
    polar.plot([avgtheta, avgtheta], [0.0, avgDSi*peakSpk], '0.')
    # polar.set_rmax(8) #radius max
    # polar.set_rticks([0.5, 1, 1.5, 2]) #radial ticks
    polar.set_rlabel_position(-22.5)  # labels away from line
    # polar.grid(True)
    polar.set_title("Spike Number; DSi = "+str(
        np.round(avgDSi, decimals=2)), va='bottom')
    plt.show()

    rho = tempRho


def dirVC():
    '''
    Similar to dirRun(), but running in voltage-clamp mode to record current at
    the soma. All other inputs are blocked when recording a particular syanptic
    input. Start script with vcPas=1 to block membrane channels prior to
    voltage-clamp experiments.
    '''
    global iOff, eOff, cPi, rho, Econ, Icon, NMcon, seed
    tempRho = rho  # store
    simultaneous = 1  # reset seeds between E and I (mimic dynamic)
    nDirs = len(dirs)
    h.progress = 0.0

    dirIoff, dirEoff, dirPi, dirRho = [], [], [], []
    for i in range(len(dirs)):
        # sigmoidal scaling of offsets, amplitudes, etc over direction
        dirIoff.append(prefIoff-(prefIoff-nullIoff)*(
            1.0 - .98/(1.0 + np.exp((dirs[i] - 74.69)/24.36))))  # santhosh
        dirEoff.append(prefEoff-(prefEoff-nullEoff)*(
            1.0 - .98/(1.0 + np.exp((dirs[i] - 91.0)/25.0))))  # sharp
        dirPi.append(prefPi+(nullPi-prefPi)*(
            1.0 - .98/(1.0 + np.exp((dirs[i] - 91.0)/25.0))))
        dirRho.append(prefRho+(nullRho-prefRho)*(
            1.0 - .98/(1.0 + np.exp((dirs[i] - 91.0)/25.0))))  # sharp

    dirCurr = list(range(nDirs))
    dirPkInh, dirArInh = [list(range(nDirs)) for i in range(2)]
    dirPkExc, dirArExc = [list(range(nDirs)) for i in range(2)]
    dirPkAmpa, dirArAmpa = [list(range(nDirs)) for i in range(2)]
    dirPkAch, dirArAch = [list(range(nDirs)) for i in range(2)]
    trialPkInh, trialArInh = [list(range(dirTrials)) for i in range(2)]
    trialPkExc, trialArExc = [list(range(dirTrials)) for i in range(2)]
    trialPkAmpa, trialArAmpa = [list(range(dirTrials)) for i in range(2)]
    trialPkAch, trialArAch = [list(range(dirTrials)) for i in range(2)]
    trialInh, trialExc = [list(range(dirTrials)) for i in range(2)]
    trialAmpa, trialAch = [list(range(dirTrials)) for i in range(2)]
    # metrics for inhibition
    PkDSi, PkTheta = [list(range(dirTrials)) for i in range(2)]
    ArDSi, ArTheta = [list(range(dirTrials)) for i in range(2)]

    if rotateMode:
        origin = findOrigin(allDends)

    # setup voltage clamp
    h('objref VC')
    VC = h.VC  # make py pointer for ease
    VC = h.SEClamp(.5)
    VC.dur1 = h.tstop  # (ms)
    VC.dur2 = 0  # just hold same for entire duration
    VC.dur3 = 0

    vecI = h.Vector()
    # note: no & and backwards + _ref_ notation in python
    vecI.record(VC._ref_i)

    # just for E and I right now (no NMDA)
    for k in range(4):  # 0 = EPSC; 1 = IPSC; 2 = ACH; 3 = AMPA
        if simultaneous:
            seed = 0
        if k == 0:  # run -60mV for EPSC (ACH + AMPA)
            VC.amp1 = -60  # (mV)
            for i in range(len(Isyn)):
                for q in range(len(Icon[i])):
                    Icon[i][q].weight[0] = 0
                    NMcon[i][q].weight[0] = 0
        elif k == 1:  # run 0mV for IPSC
            VC.amp1 = 0
            for i in range(len(Esyn)):
                for q in range(len(Econ[i])):
                    Econ[i][q].weight[0] = 0
                    AMPcon[i][q].weight[0] = 0
        elif k == 2:  # run -60mV for ACH isolated
            VC.amp1 = -60
            for i in range(len(Isyn)):
                for q in range(len(Icon[i])):
                    Icon[i][q].weight[0] = 0
                    AMPcon[i][q].weight[0] = 0
        elif k == 3:  # run -60mV for AMPA isolated
            VC.amp1 = -60
            for i in range(len(Isyn)):
                for q in range(len(Icon[i])):
                    Econ[i][q].weight[0] = 0
        # now run through trials and directions
        for j in range(dirTrials):
            for i in range(len(dirs)):
                if rotateMode:
                    dirXlocs, dirYlocs = rotate(
                        origin, xLocs, yLocs, np.radians(dirLabel[i]))
                else:
                    dirXlocs, dirYlocs = xLocs, yLocs
                    iOff = dirIoff[i]
                    eOff = dirEoff[i]

                if not nonDirectionalPi:
                    cPi = dirPi[i]
                else:
                    cPi = nullPi

                # rho = dirRho[i] # experimental
                dataRun(_xLocs=dirXlocs, _yLocs=dirYlocs,
                        vcMode=1, dirRunning=1)
                dirCurr[i] = np.array(vecI)  # store currents
            if k == 0:
                for i in range(len(dirs)):
                    epsc = dirCurr[i]
                    dirPkExc[i] = np.amax(epsc[30:])
                    dirArExc[i] = np.sum(epsc[30:]/len(epsc[30:]))
            trialPkExc[j] = dirPkExc[:]
            trialArExc[j] = dirArExc[:]
            trialExc[j] = dirCurr[:]
            if not j:
                avgExc = list(range(len(dirs)))
                for i in range(len(dirs)):
                    avgExc[i] = np.zeros(len(dirCurr[0]))
            for i in range(len(dirs)):
                avgExc[i] += dirCurr[i]
            if k == 1:
                for i in range(len(dirs)):
                    ipsc = dirCurr[i]
                    dirPkInh[i] = np.amax(ipsc[30:])
                    dirArInh[i] = np.sum(ipsc[30:]/len(ipsc[30:]))
                    # vectors with peak
                    xpts = np.multiply(dirPkInh, np.cos(np.radians(dirLabel)))
                    ypts = np.multiply(dirPkInh, np.sin(np.radians(dirLabel)))
                    xsum = sum(xpts)
                    ysum = sum(ypts)
                    PkDSi[j] = np.sqrt(xsum**2 + ysum**2)/sum(dirPkInh)
                    PkTheta[j] = np.arctan2(ysum, xsum)*180/np.pi
                    # vecrors with area
                    xpts = np.multiply(dirArInh, np.cos(np.radians(dirLabel)))
                    ypts = np.multiply(dirArInh, np.sin(np.radians(dirLabel)))
                    xsum = sum(xpts)
                    ysum = sum(ypts)
                    ArDSi[j] = np.sqrt(xsum**2 + ysum**2)/sum(dirArInh)
                    ArTheta[j] = np.arctan2(ysum, xsum)*180/np.pi
                    # store data from this trial
                trialPkInh[j] = dirPkInh[:]
                trialArInh[j] = dirArInh[:]
                trialInh[j] = dirCurr[:]
                if not j:
                    avgInh = list(range(len(dirs)))
                    for i in range(len(dirs)):
                        avgInh[i] = np.zeros(len(dirCurr[0]))
                for i in range(len(dirs)):
                    avgInh[i] += dirCurr[i]
            if k == 2:
                for i in range(len(dirs)):
                    epsc = dirCurr[i]
                    dirPkAch[i] = np.amax(epsc[30:])
                    dirArAch[i] = np.sum(epsc[30:]/len(epsc[30:]))
                trialPkAch[j] = dirPkAch[:]
                trialArAch[j] = dirArAch[:]
                trialAch[j] = dirCurr[:]
                if not j:
                    avgACH = list(range(len(dirs)))
                    for i in range(len(dirs)):
                        avgACH[i] = np.zeros(len(dirCurr[0]))
                for i in range(len(dirs)):
                    avgACH[i] += dirCurr[i]
            if k == 3:
                for i in range(len(dirs)):
                    epsc = dirCurr[i]
                    dirPkAmpa[i] = np.amax(epsc[30:])
                    dirArAmpa[i] = np.sum(epsc[30:]/len(epsc[30:]))
                trialPkAmpa[j] = dirPkAmpa[:]
                trialArAmpa[j] = dirArAmpa[:]
                trialAmpa[j] = dirCurr[:]
                if not j:
                    avgAMPA = list(range(len(dirs)))
                    for i in range(len(dirs)):
                        avgAMPA[i] = np.zeros(len(dirCurr[0]))
                for i in range(len(dirs)):
                    avgAMPA[i] += dirCurr[i]

            h.progress = h.progress + 100.0/(dirTrials*2)

        if k == 0:
            for i in range(len(Isyn)):
                for q in range(len(Icon[i])):
                    Icon[i][q].weight[0] = inhibWeight  # restore inhibition
        elif k == 1:
            for i in range(len(NMsyn)):
                for q in range(len(Econ[i])):
                    Econ[i][q].weight[0] = excWeight  # restore ACH
        elif k == 2:
            for i in range(len(AMPsyn)):
                for q in range(len(AMPcon[i])):
                    AMPcon[i][q].weight[0] = ampaWeight  # restore AMPA
        elif k == 3:
            for i in range(len(Isyn)):
                for q in range(len(Icon[i])):
                    Icon[i][q].weight[0] = inhibWeight  # restore inhibition
                    Econ[i][q].weight[0] = excWeight  # restore all excitation
                    NMcon[i][q].weight[0] = nmdaWeight

    for i in range(len(dirs)):
        avgExc[i] /= dirTrials
        avgInh[i] /= dirTrials
        avgACH[i] /= dirTrials
        avgAMPA[i] /= dirTrials

    # avg inh peak in each direction and avg DSi and theta
    for i in range(len(dirs)):
        ipsc = avgInh[i]
        dirPkInh[i] = np.amax(ipsc[30:])
        dirArInh[i] = np.sum(ipsc[30:]/len(ipsc[30:]))
    # from peak
    xpts = np.multiply(dirPkInh, np.cos(np.radians(dirLabel)))
    ypts = np.multiply(dirPkInh, np.sin(np.radians(dirLabel)))
    xsum = sum(xpts)
    ysum = sum(ypts)
    avgPkDSi = np.sqrt(xsum**2 + ysum**2)/sum(dirPkInh)
    avgPkTheta = np.arctan2(ysum, xsum)*180/np.pi
    # from area
    xpts = np.multiply(dirArInh, np.cos(np.radians(dirLabel)))
    ypts = np.multiply(dirArInh, np.sin(np.radians(dirLabel)))
    xsum = sum(xpts)
    ysum = sum(ypts)
    avgArDSi = np.sqrt(xsum**2 + ysum**2)/sum(dirArInh)
    avgArTheta = np.arctan2(ysum, xsum)*180/np.pi

    print('dirIoff:')
    print(np.round(dirIoff, decimals=3))
    print('dirPi:')
    print(np.round(dirPi, decimals=3))
    # print('dirRho:')
    # print(np.round(dirRho,decimals=3))
    print('avg peak inhibition:')
    print(np.round(dirPkInh, decimals=3))
    print('avg peak DSi: ' + str(np.round(avgPkDSi, decimals=3)))
    print('avg peak theta: ' + str(np.round(avgPkTheta, decimals=2)))
    print('DSis (peak):')
    print(np.round(PkDSi, decimals=3))
    print('	sdev: ' + str(np.round(np.std(PkDSi), decimals=2)))
    print('thetas (peak):')
    print(np.round(PkTheta, decimals=2))
    print('	sdev: ' + str(np.round(np.std(PkTheta), decimals=2)))
    print('avg area of inhibition:')
    print(np.round(dirArInh, decimals=3))
    print('avg area DSi: ' + str(np.round(avgArDSi, decimals=3)))
    print('avg area theta: ' + str(np.round(avgArTheta, decimals=2)))
    print('DSis (area):')
    print(np.round(ArDSi, decimals=3))
    print('	sdev: ' + str(np.round(np.std(ArDSi), decimals=2)))
    print('thetas (peak):')
    print(np.round(ArTheta, decimals=2))
    print('	sdev: ' + str(np.round(np.std(ArTheta), decimals=2)))

    # avg inhibition and excitation for each direction
    dirAvgExcDF = pd.DataFrame(np.array(avgExc).T, columns=dirLabel)
    dirAvgExcDF.to_hdf(basest+'dirVC.h5', key='dirAvgExc', mode='w')
    dirAvgInhDF = pd.DataFrame(np.array(avgExc).T, columns=dirLabel)
    dirAvgInhDF.to_hdf(basest+'dirVC.h5', key='dirAvgInh', mode='a')
    dirAvgAchDF = pd.DataFrame(np.array(avgACH).T, columns=dirLabel)
    dirAvgAchDF.to_hdf(basest+'dirVC.h5', key='dirAvgACH', mode='a')
    dirAvgAmpaDF = pd.DataFrame(np.array(avgAMPA).T, columns=dirLabel)
    dirAvgAmpaDF.to_hdf(basest+'dirVC.h5', key='dirAvgAMPA', mode='a')
    # inh and exc traces for every direction of every trial
    trialInh = np.array(trialInh).T.reshape(len(avgInh[0]), -1)
    mi = pd.MultiIndex.from_product(
        [np.arange(dirTrials), dirLabel], names=('trials', 'direction'))
    trialInhDF = pd.DataFrame(trialInh, columns=mi)
    trialInhDF.to_hdf(basest+'dirVC.h5', key='trialInh', mode='a')
    trialExc = np.array(trialExc).T.reshape(len(avgExc[0]), -1)
    trialExcDF = pd.DataFrame(trialExc, columns=mi)
    trialExcDF.to_hdf(basest+'dirVC.h5', key='trialExc', mode='a')
    trialAch = np.array(trialAch).T.reshape(len(avgACH[0]), -1)
    trialAchDF = pd.DataFrame(trialAch, columns=mi)
    trialAchDF.to_hdf(basest+'dirVC.h5', key='trialACH', mode='a')
    trialAmpa = np.array(trialAmpa).T.reshape(len(avgAMPA[0]), -1)
    trialAmpaDF = pd.DataFrame(trialAmpa, columns=mi)
    trialAmpaDF.to_hdf(basest+'dirVC.h5', key='trialAMPA', mode='a')
    # peak and area for inhibition and excitation
    trialPkInhDF = pd.DataFrame(np.array(trialPkInh).T, index=dirLabel)
    trialPkInhDF.to_hdf(basest+'dirVC.h5', key='trialPkInh', mode='a')
    trialArInhDF = pd.DataFrame(np.array(trialArInh).T, index=dirLabel)
    trialArInhDF.to_hdf(basest+'dirVC.h5', key='trialArInh', mode='a')
    trialPkExcDF = pd.DataFrame(np.array(trialPkExc).T, index=dirLabel)
    trialPkExcDF.to_hdf(basest+'dirVC.h5', key='trialPkExc', mode='a')
    trialArExcDF = pd.DataFrame(np.array(trialArExc).T, index=dirLabel)
    trialArExcDF.to_hdf(basest+'dirVC.h5', key='trialArExc', mode='a')
    trialPkAchDF = pd.DataFrame(np.array(trialPkAch).T, index=dirLabel)
    trialPkAchDF.to_hdf(basest+'dirVC.h5', key='trialPkACH', mode='a')
    trialArAchDF = pd.DataFrame(np.array(trialArAch).T, index=dirLabel)
    trialArAchDF.to_hdf(basest+'dirVC.h5', key='trialArACH', mode='a')
    trialPkAmpaDF = pd.DataFrame(np.array(trialPkAmpa).T, index=dirLabel)
    trialPkAmpaDF.to_hdf(basest+'dirVC.h5', key='trialPkAMPA', mode='a')
    trialArAmpaDF = pd.DataFrame(np.array(trialArAmpa).T, index=dirLabel)
    trialArAmpaDF.to_hdf(basest+'dirVC.h5', key='trialArAMPA', mode='a')

    plt.figure(1)
    time = np.float64(range(int(h.tstop/h.dt)+1))*h.dt
    for i in range(len(dirs)):
        epsc = avgExc[i]
        avgExc[i] -= sum(epsc[30:59])/30.0
        ipsc = avgInh[i]
        avgInh[i] -= sum(ipsc[30:59])/30.0
    plt.plot(time[30:], epsc[30:], 'r')
    plt.plot(time[30:], ipsc[30:], 'b')
    plt.xlabel('time (ms)')
    plt.ylabel('current')

    # make polar plot with all trials (peak)
    plt.figure(2)
    PkTheta = np.deg2rad(PkTheta)
    polar1 = plt.subplot(111, projection='polar')
    peakInh = 0
    for i in range(dirTrials):
        if max(trialPkInh[i]) > peakInh:
            peakInh = max(trialPkInh[i])
            circInh = np.array(trialPkInh[i])
            circInh = circInh[inds]
            circInh = np.append(circInh, circInh[0])
            polar1.plot(circle, circInh, '.75')
    for i in range(dirTrials):
        # DSi is on a scale of 0 to 1
        polar1.plot([PkTheta[i], PkTheta[i]], [0.0, PkDSi[i]*peakInh], '.75')
    # add in avg values
    circInh = np.array(dirPkInh)
    circInh = circInh[inds]
    circInh = np.append(circInh, circInh[0])
    avgPkTheta = np.deg2rad(avgPkTheta)
    polar1.plot(circle, circInh, '0.')
    polar1.plot([avgPkTheta, avgPkTheta], [0.0, avgPkDSi*peakInh], '0.')
    # polar1.set_rmax(8) #radius max
    # polar1.set_rticks([0.5, 1, 1.5, 2]) #radial ticks
    polar1.set_rlabel_position(-22.5)  # labels away from line
    # polar1.grid(True)
    polar1.set_title("IPSC peak; DSi = "+str(
        np.round(avgPkDSi, decimals=2)), va='bottom')

    # make polar plot with all trials (area)
    plt.figure(3)
    ArTheta = np.deg2rad(ArTheta)
    polar2 = plt.subplot(111, projection='polar')
    peakInh = 0
    for i in range(dirTrials):
        if max(trialArInh[i]) > peakInh:
            peakInh = max(trialArInh[i])
        circInh = np.array(trialArInh[i])
        circInh = circInh[inds]
        circInh = np.append(circInh, circInh[0])
        polar2.plot(circle, circInh, '.75')
    for i in range(dirTrials):
        # DSi is on a scale of 0 to 1
        polar2.plot([ArTheta[i], ArTheta[i]], [0.0, ArDSi[i]*peakInh], '.75')
    # add in avg values
    circInh = np.array(dirArInh)
    circInh = circInh[inds]
    circInh = np.append(circInh, circInh[0])
    avgArTheta = np.deg2rad(avgArTheta)
    polar2.plot(circle, circInh, '0.')
    polar2.plot([avgArTheta, avgArTheta], [0.0, avgArDSi*peakInh], '0.')
    # polar2.set_rmax(8) #radius max
    # polar2.set_rticks([0.5, 1, 1.5, 2]) #radial ticks
    polar2.set_rlabel_position(-22.5)  # labels away from line
    # polar2.grid(True)
    polar2.set_title("IPSC area; DSi = "+str(
        np.round(avgArDSi, decimals=2)), va='bottom')

    plt.show()  # display all figures

    rho = tempRho  # reset
