# -*- coding: utf-8 -*-
"""
TANK SIZING OPTIMISATION MODEL (SCENARIO STOCH) + ORIFICE HEIGHT V2
==============================
21 Oct 2021 Qiao Yan Soh

Tank parameters are discretized, with fixed orifice areas and heights. 
Finds an optimal system size for a singular given rainfall and demand. 

"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd 


def CostObjFx(m):
    """ Minimising costs associated with using freshwater top ups. """
    return sum(m.CostF[t, S] for t in m.t for S in m.S)

# ----- Tank design constraints 
def AreaFx(m, j):
    return m.TankArea[j] == sum(m.a[j, p] * m.ABinary[j, p] for p in m.p)

def HeightFx(m, j):
    return m.TankHeight[j] == sum(m.h[j, q] * m.HBinary[j, q] for q in m.q)

def AreaBinary(m, j): 
    return sum(m.ABinary[j, p] for p in m.p) == 1.0

def HeightBinary(m, j): 
    return sum(m.HBinary[j , q] for q in m.q) == 1.0

# ----- Volume variable definition for linearisation 
def VolDefinition(m, j, k, S):
    return m.Volume[j, k, S] == sum(m.a[j, p] * m.V[j, p, k, S] for p in m.p)

def VolDummyFx1(m, j, p, k, S):
    return m.V[j, p, k, S] <= m.ABinary[j, p] * m.HUB

def VolDummyFx2(m, j, p, k, S):
    return m.V[j, p, k, S] <= m.Level[j, k, S]

def VolDummyFx3(m, j, p, k, S):
    return m.V[j, p, k, S] >= m.Level[j, k, S] - (1 - m.ABinary[j, p]) * m.HUB

# ----- Tank capacity definition 
def VolMaxDefinition(m, j):
    return m.MaxVol[j] == sum(m.a[j, p] * m.h[j, q] * m.v[j, p, q] for p in m.p for q in m.q)

def MaxVolDummyFx1(m, j, p, q):
    return m.v[j, p, q] <= m.ABinary[j, p]

def MaxVolDummyFx2(m, j, p, q):
    return m.v[j, p, q] <= m.HBinary[j, q]

def MaxVolDummyFx3(m, j, p, q):
    return m.v[j, p, q] >= m.ABinary[j, p] + m.HBinary[j, q] - 1

# ----- Orifice design constraints 
def OrificeHeightFx1(m, c, ):
    if c in ['SH', 'SD2']:
        return m.OrificeHeight[c] == sum(m.OH[c, n] * m.OHDummy1[c, n, 'S'] for n in m.n)
    else:
        return m.OrificeHeight[c] == sum(m.OH[c, n] * m.OHDummy1[c, n, 'D'] for n in m.n)

def OrificeHeightFx2(m, c, n, j):
    return m.OHDummy1[c, n, j] <= m.OHBinary[c, n] * 100

def OrificeHeightFx3(m, c, n, j):
    return m.OHDummy1[c, n, j] <= m.TankHeight[j]

def OrificeHeightFx4(m, c, n, j):
    return m.OHDummy1[c, n, j] >= m.TankHeight[j] - (1 - m.OHBinary[c, n]) * 100 

    
def OrificeHeightBinary(m, c):
    return sum(m.OHBinary[c, n] for n in m.n) == 1.0

def SepOrificeLimit(m, ):
    return m.OrificeHeight['SD2'] >= m.OrificeHeight['SH'] + 0.125 # smallest possible increment = 0.1 * 1.25 

# ----- Tank mass balances 
def SepBalanceFx(m, t, S):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t+1, S] + m.Volume['S', t+1, S] == m.Volume['S', t, S] + m.RainIn[t, S] + m.Overflow[t, S] \
            - m.Discharge['SH', t, S] - m.Discharge['SD', t, S] - m.Discharge['SD2', t, S]
            
def DetBalanceFx(m, t, S):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Volume['D', t+1, S] == m.Volume['D', t, S] + m.Discharge['SD', t, S] + m.Discharge['SD2', t, S] \
            - m.Discharge['DO', t, S] - m.Discharge['DO2', t, S]
            
def HarBalanceFx(m, t, S):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Volume['H', t+1, S] == m.Volume['H', t, S] + m.Discharge['SH', t, S] - m.Discharge['HT', t, S]
    
def TrtBalanceFx(m, t, S):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Volume['T', t+1, S] == m.Volume['T', t, S] + m.Discharge['HT', t, S] - m.Discharge['TD', t, S]
    
# ----- Freshwater 
def FreshwaterFx(m, t, S):
    """ Freshwater top up for demand. """
    return m.Demand[t] == m.Freshwater[t, S] + m.Discharge['TD', t, S]             

def FreshwaterCostFx(m, t, S):
    """ Cost of using freshwater to top up for demand. """
    return m.CostF[t, S] == m.Freshwater[t, S] * m.FreshwaterCost[t]


# ----- Overflow constraints 
def OverflowFx(m, t, S):
    # if t == m.t[-1]:
    #     return pyo.Constraint.Skip
    # else:
    #     return m.Overflow[t+1, S] >= m.Volume['S', t, S] + m.RainIn[t, S] + m.Overflow[t, S] \
    #         - m.Discharge['SH', t, S] - m.Discharge['SD', t, S] - m.Discharge['SD2', t, S] - m.MaxVol['S']
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t, S] >= m.Volume['S', t, S] + m.RainIn[t, S] + m.Overflow[t-1, S] \
            - m.Discharge['SH', t, S] - m.Discharge['SD', t, S] - m.Discharge['SD2', t, S] - m.MaxVol['S']

def OverflowFXUB(m, t, S):
    # if t == m.t[-1]:
    #     return pyo.Constraint.Skip
    # else:
    #     return m.Overflow[t+1, S] <= m.Volume['S', t, S] + m.RainIn[t, S] + m.Overflow[t, S] \
    #         - m.Discharge['SH', t, S] - m.Discharge['SD', t, S] - m.Discharge['SD2', t, S] - m.MaxVol['S'] \
    #             - 5000 * m.OverflowBinary[t, S] + 5000
 
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t, S] <= m.Volume['S', t, S] + m.RainIn[t, S] + m.Overflow[t-1, S] \
            - m.Discharge['SH', t, S] - m.Discharge['SD', t, S] - m.Discharge['SD2', t, S] - m.MaxVol['S'] \
                - 5000 * m.OverflowBinary[t, S] + 5000
                
def OverflowLm(m, t, S):
    """ Desired upper bound for overflow. """
    return m.Overflow[t, S] <= m.Epsilon * m.OverflowBinary[t, S]

# ----- Conditional Flows
def CondDummyFx(m, C, q, t, S):
    return m.CondDummy[C, q, t, S] <= m.CondBinary[C, t, S]

def CondDummyFx2(m, C, q, t, S):
    if C in ['SH', 'SD2']:
        return m.CondDummy[C, q, t, S] <= m.HBinary['S', q]
    elif C in ['DO2']:
        return m.CondDummy[C, q, t, S] <= m.HBinary['D', q]

def CondDummyFx3(m, C, q, t, S):
    if C in ['SH', 'SD2']:
        return m.CondDummy[C, q, t, S] >= m.CondBinary[C, t, S] + m.HBinary['S', q] - 1
    elif C in ['DO2']:
        return m.CondDummy[C, q, t, S] >= m.CondBinary[C, t, S] + m.HBinary['D', q] - 1


def NDummyFx(m, c, n, k, s, j):
    return m.NDummy[c, n, k, s, j] <= m.OHDummy1[c, n, j]

def NDummyFx2(m, c, n, k, s, j):
    return m.NDummy[c, n, k, s, j] >= m.OHDummy1[c, n, j] - (1 - m.CondBinary[c, k, s]) * 1000

def NDummyFx3(m, c, n, k, s, j):
    return m.NDummy[c, n, k, s, j] <= m.CondBinary[c, k, s] * 1000


# ----- 
def SHLowerFx(m, t, S):
    return m.Discharge['SH', t, S] >= 2.66 * m.OrificeArea['SH'] * (m.Level['S', t, S] - m.OrificeHeight['SH'])

def SHUpperFx1(m, t, S):
    return m.Discharge['SH', t, S] <= 2.66 * m.OrificeArea['SH'] * m.DeltaT \
        * (sum(m.h['S', q] * m.CondDummy['SH', q, t, S] for q in m.q) - sum(m.OH['SH', n] * m.NDummy['SH', n, t, S, 'S'] for n in m.n))

def SHUpperFx2(m, t, S):
    """ Sets upper bound when condition is True. """
    return m.Discharge['SH', t, S] <= 2.66 * m.OrificeArea['SH'] * m.DeltaT * (m.Level['S', t, S] - m.OrificeHeight['SH']) \
        - 2.66 * m.OrificeArea['SH'] * m.DeltaT * sum(m.OH['SH', n] * m.NDummy['SH', n, t, S, 'S'] for n in m.n) \
            + 2.66 * m.OrificeArea['SH'] * m.DeltaT * m.OrificeHeight['SH']

# -----             
def SD2LowerFx(m, t, S):
    """ Lower bound for SD2, effective when condition is True."""
    return m.Discharge['SD2', t, S] >= 2.66 * m.OrificeArea['SD2'] * (m.Level['S', t, S] - m.OrificeHeight['SD2']) # * m.DeltaT

def SD2UpperFx1(m, t, S):
    """ Sets upper bound of 0 for SD2 flow when conditions are not met """
    return m.Discharge['SD2', t, S] <= 2.66 * m.OrificeArea['SD2'] * m.DeltaT \
        * (sum(m.h['S', q] * m.CondDummy['SD2', q, t, S] for q in m.q) - sum(m.OH['SD2', n] * m.NDummy['SD2', n, t, S, 'S'] for n in m.n))
        
def SD2UpperFx2(m, t, S): 
    """ Sets upper bound when condition is True. """
    return m.Discharge['SD2', t, S] <= 2.66 * m.OrificeArea['SD2'] * m.DeltaT * (m.Level['S', t, S] - m.OrificeHeight['SD2']) \
        - 2.66 * m.OrificeArea['SD2'] * m.DeltaT * sum(m.OH['SD2', n] * m.NDummy['SD2', n, t, S, 'S'] for n in m.n) \
            + 2.66 * m.OrificeArea['SD2'] * m.DeltaT * m.OrificeHeight['SD2']
            
# ----- 
def DO2LowerFx(m, t, S):
    """ """
    return m.Discharge['DO2', t, S] >= 2.66 * m.OrificeArea['DO2'] * (m.Level['D', t, S] - m.OrificeHeight['DO2'])

def DO2UpperFx1(m, t, S):
    """ Sets upper bound of 0 for SD2 flow when conditions are not met """
    return m.Discharge['DO2', t, S] <= 2.66 * m.OrificeArea['DO2'] * m.DeltaT  \
        * (sum(m.h['D', q] * m.CondDummy['DO2', q, t, S] for q in m.q) - sum(m.OH['DO2', n] * m.NDummy['DO2', n, t, S, 'D'] for n in m.n))
        
def DO2UpperFx2(m, t, S): 
    """ Sets upper bound when condition is True. """
    return m.Discharge['DO2', t, S] <= 2.66 * m.OrificeArea['DO2'] * m.DeltaT * (m.Level['D', t, S] - m.OrificeHeight['DO2']) \
        - 2.66 * m.OrificeArea['DO2'] * m.DeltaT * sum(m.OH['DO2', n] * m.NDummy['DO2', n, t, S, 'D'] for n in m.n) \
            + 2.66 * m.OrificeArea['DO2'] * m.DeltaT * m.OrificeHeight['DO2']

# ----- Unconditional Flows 
def DOFlowFx(m, t, S):
    """ Inflexible, unconditional flow out of the system. """
    return m.Discharge['DO', t, S] <= 2.66 * m.OrificeArea['DO'] * m.Level['D', t, S] * m.DeltaT

def SDFlowFx(m, t, S):
    """ """
    return m.Discharge['SD', t, S] <= 2.66 * m.OrificeArea['SD'] * m.Level['S', t, S] * m.DeltaT

# ----- 
def RelayBinariesFX(m, t, S):
    """ """
    return m.RelayBinary[1, t, S] + m.RelayBinary[2, t, S] + m.RelayBinary[3, t, S] == 1
    
def HTRelay1(m, t, S): 
    """ """
    return m.Discharge['HT', t, S] >= m.PumpRate * m.RelayBinary[1, t, S]

def HTRelay2(m, t, S):
    """ """
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.Discharge['HT', t, S] <= m.Discharge['HT', t-1, S] - m.PumpRate * m.RelayBinary[2, t, S] + m.PumpRate

def HTRelay3(m, t, S):
    """ """
    return m.Discharge['HT', t, S] <= (1 - m.RelayBinary[3, t, S]) * m.PumpRate

def HTRelay4(m, t, S):
    """ """
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.Discharge['HT', t, S] >= m.Discharge['HT', t-1, S] + m.PumpRate * m.RelayBinary[2, t, S] - m.PumpRate

def HTRelay5(m, t, S):
    """ """
    return m.Level['H', t, S] >= m.HTOnPt * m.RelayBinary[1, t, S] + m.HTOffPt * m.RelayBinary[2, t, S]

def HTRelay6(m, t, S):
    """ """
    return m.Level['H', t, S] <= sum(m.h['H', q] * m.RelayDummy[q, t, S] for q in m.q) \
        + m.HTOnPt * m.RelayBinary[2, t, S] + m.HTOffPt * m.RelayBinary[3, t, S]

def RelayDummyFx1(m, q, t, S):
    return m.RelayDummy[q, t, S] <= m.RelayBinary[1, t, S]

def RelayDummyFx2(m, q, t, S):
    return m.RelayDummy[q, t, S] <= m.HBinary['H', q]

def RelayDummyFx3(m, q, t, S):
    return m.RelayDummy[q, t, S] >= m.RelayBinary[1, t, S] + m.HBinary['H', q] - 1

# ----- Vol Cons
def VolConsFx(m, t, S):
    """ Overall volume conservation of the system. """
    return m.Overflow[t, S] + m.Volume['S', t, S] + m.Volume['H', t, S] + m.Volume['D', t, S] \
        == m.Volume['S', 0, S] + m.Volume['H', 0, S] + m.Volume['D', 0, S] \
                + sum(m.RainIn[k, S] for k in range(0, t)) - sum(m.Discharge['DO', k, S] for k in range(0, t)) \
                    - sum(m.Discharge['HT', k, S] for k in range(0, t)) - sum(m.Discharge['DO2', k, S] for k in range(0, t))

# ----- system Behavioural constraints 
def EndFx(m, S):
    """ Leaves separation tank levels to a desired level at the end of the optimisation loop """
    return m.Level['S', m.t[-1], S] <= m.PrpEndLevel * m.TankHeight['S']

def FirstFlowFX(m, t, S):
    """ SH at t=0 is set with initialisation. """
    x = 48 - t
    checkerLB = t - 48 + x
    checker = sum(m.RainIn[k, S] for k in range(checkerLB, t))
    if ((m.RainIn[t, S] != 0) & (checker == 0)):
        return m.Discharge['SH', t, S] == 0.0
    else:
        return pyo.Constraint.Skip 
    
# ----- Variable Initialisation
def LevelInitFx(m, T, S):
    """ Initializes Level variables """
    return m.Level[T, 0, S] == m.InitialLevel[T]

def DischargeInitFx(m, O, S):
    """ Initializes Discharge variables """
    return m.Discharge[O, 0, S] == 0.0 

def OverflowInitFx(m, S):
    """ Initializes Overflow variable """
    return m.Overflow[0, S] == 0.0

def InitRelayFX(m, S): 
    return m.RelayBinary[3, 0, S] == 1


# ----- Variable Bound Assignment   
def LevelBounds(m, T, t, S): 
    """ Sets upper bounds for Level variables """
    return m.Level[T, t, S] <= m.TankHeight[T]

def DischargeBounds(m, O, t, S):
    """ Sets upper bounds for Discharge variables """
    return m.Discharge[O, t, S] <= m.DischargeLimit[O]

def TotalVolumeLimit(m):
    return sum(m.MaxVol[T] for T in m.Tanks) <= 700.0


def DischargeLimitsFx(m, O):
    if O in ['SH', 'SD2']:
        return m.DischargeLimit[O] == 2.66 * m.OrificeArea[O] * (m.TankHeight['S'] - m.OrificeHeight[O]) * m.DeltaT
    elif O == 'SD':
        return m.DischargeLimit[O] == 2.66 * m.OrificeArea[O] * m.TankHeight['S'] * m.DeltaT
    elif O == 'DO':
        return m.DischargeLimit[O] == 2.66 * m.OrificeArea[O] *  m.TankHeight['D'] * m.DeltaT
    elif O == 'HT':
        return m.DischargeLimit[O] == 5/12
    elif O == 'TD':
        return m.DischargeLimit[O] == 500
    elif O == 'DO2':
        return m.DischargeLimit[O] ==  2.66 * m.OrificeArea[O] * (m.TankHeight['D'] - m.OrificeHeight[O]) * m.DeltaT

def PUBDischargeLim(m, t, S):
    return m.Discharge['DO2', t, S] + m.Discharge['DO', t, S] <= 156.3

#%% 
def CreateAbstractModel(DeltaT = 300, Epsilon = 100, PrpEndLevel = 0.8):
    m = pyo.AbstractModel()
    
    # ===== SETS 
    m.t = pyo.Set(ordered = True, doc = 'Simulation Timesteps')
    m.S = pyo.Set(ordered = True, doc = 'Rainfall Scenario index')
    m.Tanks = pyo.Set()
    m.Outlets = pyo.Set()
    m.CondOutlets = pyo.Set(within = m.Outlets)
    
    m.p = pyo.Set(doc = 'Tank Area Options')
    m.q = pyo.Set(doc = 'Tank Height Options')
    m.n = pyo.Set(doc = 'Orifice Height Options')
    
    m.r = pyo.Set(initialize = [1,2,3],  doc = 'Relay binary')
    
    # ===== PARAMETERS 
    m.a = pyo.Param(m.Tanks, m.p, within = pyo.NonNegativeReals)
    m.h = pyo.Param(m.Tanks, m.q, within = pyo.NonNegativeReals)
    
    m.RainIn = pyo.Param(m.t, m.S, within = pyo.NonNegativeReals)
    m.Demand = pyo.Param(m.t, within = pyo.NonNegativeReals)
    m.FreshwaterCost = pyo.Param(m.t, within = pyo.NonNegativeReals)
    m.OrificeArea = pyo.Param(m.Outlets, within = pyo.NonNegativeReals)
    m.OH = pyo.Param(m.CondOutlets, m.n, within = pyo.NonNegativeReals)
    m.InitialLevel = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    
    # ===== SCALARS 
    m.DeltaT = DeltaT
    m.Epsilon = Epsilon 
    m.HUB = 50.0
    m.PrpEndLevel = PrpEndLevel 
    
    m.HTOnPt = 1.0
    m.HTOffPt = 0.3
    m.PumpRate = 5/12
    
    # ===== VARIABLES
    m.ABinary = pyo.Var(m.Tanks, m.p, within = pyo.Binary)
    m.HBinary = pyo.Var(m.Tanks, m.q, within = pyo.Binary)
    m.CondBinary = pyo.Var(m.CondOutlets, m.t, m.S, within = pyo.Binary)
    m.OverflowBinary = pyo.Var(m.t, m.S, within = pyo.Binary)
    m.RelayBinary = pyo.Var(m.r, m.t, m.S, within = pyo.Binary)
    m.OHBinary = pyo.Var(m.CondOutlets, m.n, within = pyo.Binary)
    
    m.TankArea = pyo.Var(m.Tanks, within = pyo.NonNegativeReals)
    m.TankHeight = pyo.Var(m.Tanks, within = pyo.NonNegativeReals)
    m.OrificeHeight = pyo.Var(m.CondOutlets, within = pyo.NonNegativeReals)
    m.MaxVol = pyo.Var(m.Tanks, within = pyo.NonNegativeReals)
    
    m.Volume = pyo.Var(m.Tanks, m.t, m.S, within = pyo.NonNegativeReals)
    m.Level = pyo.Var(m.Tanks, m.t, m.S, within = pyo.NonNegativeReals)
    m.Overflow = pyo.Var(m.t, m.S, within = pyo.NonNegativeReals)
    m.Discharge = pyo.Var(m.Outlets, m.t, m.S, within = pyo.NonNegativeReals)
    m.Freshwater = pyo.Var(m.t, m.S, within = pyo.NonNegativeReals)
    
    m.CostF = pyo.Var(m.t, m.S, within = pyo.NonNegativeReals)
    m.DischargeLimit = pyo.Var(m.Outlets, within = pyo.NonNegativeReals)
    
    m.V = pyo.Var(m.Tanks, m.p, m.t, m.S, within = pyo.NonNegativeReals, doc = 'Volume calculation dummy variable')
    m.v = pyo.Var(m.Tanks, m.p, m.q, within = pyo.NonNegativeReals, doc = 'Max Vol Dummy')
    m.CondDummy = pyo.Var(m.CondOutlets, m.q, m.t, m.S, within = pyo.NonNegativeReals)
    m.RelayDummy = pyo.Var(m.q, m.t, m.S, within = pyo.NonNegativeReals)
    m.NDummy = pyo.Var(m.CondOutlets, m.n, m.t, m.S, m.Tanks, within = pyo.NonNegativeReals)
    m.OHDummy1 = pyo.Var(m.CondOutlets, m.n, m.Tanks, within = pyo.NonNegativeReals)
    
    # ===== EQUATIONS 
    m.Objective = pyo.Objective(rule = CostObjFx, sense = pyo.minimize)
    
    m.Area = pyo.Constraint(m.Tanks, rule = AreaFx)
    m.Height = pyo.Constraint(m.Tanks, rule = HeightFx)
    m.SelectOneArea = pyo.Constraint(m.Tanks, rule = AreaBinary)
    m.SelectOneHeight = pyo.Constraint(m.Tanks, rule = HeightBinary)
    
    m.OHeights = pyo.Constraint(m.CondOutlets, rule = OrificeHeightFx1)
    m.OHeights1 = pyo.Constraint(m.CondOutlets, m.n, m.Tanks, rule = OrificeHeightFx2)
    m.OHeights2 = pyo.Constraint(m.CondOutlets, m.n, m.Tanks, rule = OrificeHeightFx3)
    m.OHeights3 = pyo.Constraint(m.CondOutlets, m.n, m.Tanks, rule = OrificeHeightFx4)
    m.SelectOneOH = pyo.Constraint(m.CondOutlets, rule = OrificeHeightBinary)
    m.SepOrificeHeights = pyo.Constraint(rule = SepOrificeLimit)
    
    m.VolCalc = pyo.Constraint(m.Tanks, m.t, m.S, rule = VolDefinition)
    m.VolumeDummy1 = pyo.Constraint(m.Tanks, m.p, m.t, m.S, rule = VolDummyFx1)
    m.VolumeDummy2 = pyo.Constraint(m.Tanks, m.p, m.t, m.S, rule = VolDummyFx2)
    m.VolumeDummy3 = pyo.Constraint(m.Tanks, m.p, m.t, m.S, rule = VolDummyFx3)
    
    m.MaxVolCalc = pyo.Constraint(m.Tanks, rule = VolMaxDefinition)
    m.MaxVolDummy1 = pyo.Constraint(m.Tanks, m.p, m.q, rule = MaxVolDummyFx1)
    m.MaxVolDummy2 = pyo.Constraint(m.Tanks, m.p, m.q, rule = MaxVolDummyFx2)
    m.MaxVolDummy3 = pyo.Constraint(m.Tanks, m.p, m.q, rule = MaxVolDummyFx3)
    
    # ----- Mass balances
    m.SepTank = pyo.Constraint(m.t, m.S, rule = SepBalanceFx)
    m.HarTank = pyo.Constraint(m.t, m.S, rule = HarBalanceFx)
    m.DetTank = pyo.Constraint(m.t, m.S, rule = DetBalanceFx)
    m.TrtTank = pyo.Constraint(m.t, m.S, rule = TrtBalanceFx)
    
    # ----- Demand
    m.FTopUp = pyo.Constraint(m.t, m.S, rule = FreshwaterFx)
    m.FCost = pyo.Constraint(m.t, m.S, rule = FreshwaterCostFx)
    
    # ----- Overflow
    m.OF = pyo.Constraint(m.t, m.S, rule = OverflowFx)
    m.OFUB = pyo.Constraint(m.t, m.S, rule = OverflowFXUB)
    m.OFLim = pyo.Constraint(m.t, m.S, rule = OverflowLm)
    
    # ----- Flows 
    m.CondFlowDummy1 = pyo.Constraint(m.CondOutlets, m.q, m.t, m.S, rule = CondDummyFx)
    m.CondFlowDummy2 = pyo.Constraint(m.CondOutlets, m.q, m.t, m.S, rule = CondDummyFx2)
    m.CondFlowDummy3 = pyo.Constraint(m.CondOutlets, m.q, m.t, m.S, rule = CondDummyFx3)
    
    m.NDummy1 = pyo.Constraint(m.CondOutlets, m.n, m.t, m.S, m.Tanks, rule = NDummyFx)
    m.NDummy2 = pyo.Constraint(m.CondOutlets, m.n, m.t, m.S, m.Tanks, rule = NDummyFx2)
    m.NDummy3 = pyo.Constraint(m.CondOutlets, m.n, m.t, m.S, m.Tanks, rule = NDummyFx3)
    
    #m.SH1 = pyo.Constraint(m.t, rule = SHLowerFx)
    m.SH2 = pyo.Constraint(m.t, m.S, rule = SHUpperFx1)
    m.SH3 = pyo.Constraint(m.t, m.S, rule = SHUpperFx2)
    
    #m.SD21 = pyo.Constraint(m.t, m.S, rule = SD2LowerFx)
    m.SD22 = pyo.Constraint(m.t, m.S, rule = SD2UpperFx1)
    m.SD23 = pyo.Constraint(m.t, m.S, rule = SD2UpperFx2)
    
    #m.DO21 = pyo.Constraint(m.t, m.S, rule = DO2LowerFx)
    m.DO22 = pyo.Constraint(m.t, m.S, rule = DO2UpperFx1)
    m.DO23 = pyo.Constraint(m.t, m.S, rule = DO2UpperFx2)
    
    m.DO = pyo.Constraint(m.t, m.S, rule = DOFlowFx)
    m.SD = pyo.Constraint(m.t, m.S, rule = SDFlowFx)
    
    m.RB = pyo.Constraint(m.t, m.S, rule = RelayBinariesFX)
    m.HT1 = pyo.Constraint(m.t, m.S, rule = HTRelay1)
    m.HT2 = pyo.Constraint(m.t, m.S, rule = HTRelay2)
    m.HT3 = pyo.Constraint(m.t, m.S, rule = HTRelay3)
    m.HT4 = pyo.Constraint(m.t, m.S, rule = HTRelay4)
    m.HT5 = pyo.Constraint(m.t, m.S, rule = HTRelay5)
    m.HT6 = pyo.Constraint(m.t, m.S, rule = HTRelay6)
    
    m.RelayDummy1 = pyo.Constraint(m.q, m.t, m.S, rule = RelayDummyFx1)
    m.RelayDummy2 = pyo.Constraint(m.q, m.t, m.S, rule = RelayDummyFx2)
    m.RelayDummy3 = pyo.Constraint(m.q, m.t, m.S, rule = RelayDummyFx3)
    
    # ----- 
    m.VolumeConservation = pyo.Constraint(m.t, m.S, rule = VolConsFx)
    m.EndState = pyo.Constraint(m.S, rule = EndFx)
    #m.FirstFlow = pyo.Constraint(m.t, m.S, rule = FirstFlowFX)
    
    m.RelayInit = pyo.Constraint(m.S, rule = InitRelayFX)
    m.InitLevel = pyo.Constraint(m.Tanks, m.S, rule = LevelInitFx)
    m.InitDischarge = pyo.Constraint(m.Outlets, m.S, rule = DischargeInitFx)
    m.InitOverflow = pyo.Constraint(m.S, rule = OverflowInitFx)
    
    m.LevelLim = pyo.Constraint(m.Tanks, m.t, m.S, rule = LevelBounds)
    m.DischargeLim = pyo.Constraint(m.Outlets, m.t, m.S, rule = DischargeBounds)
    m.VolLim = pyo.Constraint(rule = TotalVolumeLimit)
    m.PUBLim = pyo.Constraint(m.t, m.S, rule = PUBDischargeLim)
    m.DischargeLimitsCalc = pyo.Constraint(m.Outlets, rule = DischargeLimitsFx)
    return m

#%% 

def CompileA():
    # Sep = [2.5, 5, 10, 15, 20]
    # Har = [10, 20, 40, 80, 100]
    # Det = [50, 100, 150, 200, 300]
    # Trt = [2, 4, 8, 10, 20]
    
    Sep = [5, 10, 20, 40, 60, 80, 100, 120]
    Har = [5, 10, 20, 40, 60, 80, 100, 120]
    Det = [5, 10, 20, 40, 60, 80, 100, 120]
    Trt = [5, 10, 20, 40, 60, 80, 100, 120]
    
    
    df = pd.DataFrame({'S': Sep, 'H': Har, 'D': Det, 'T': Trt})
    df = df.unstack()
    
    return dict(df)

def CompileH():
    # Sep = [0.75, 1, 1.5, 2, 2.5]
    # Har = [1.25, 1.5, 2, 2.5, 3]
    # Det = [2, 2.25, 3, 3.25, 4]
    # Trt = [0.5, 0.75, 1, 1.25, 1.5]

    Sep = [1.25, 1.5, 2, 2.5, 3]
    Har = [1.25, 1.5, 2, 2.5, 3]
    Det = [1.25, 1.5, 2, 2.5, 3]
    Trt = [1.25, 1.5, 2, 2.5, 3]
    
    df = pd.DataFrame({'S': Sep, 'H': Har, 'D': Det, 'T': Trt})
    df = df.unstack()
    
    return dict(df)

def CompileOH():
    SH = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    SD2 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    DO2 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    df = pd.DataFrame({'SH': SH, 'SD2': SD2, 'DO2': DO2})
    df = df.unstack()
    
    return dict(df)

def CompileDataParams(RainArray, DemandArray, Cost, n_scenarios, A = CompileA(), H = CompileH(), OH = CompileOH(), \
                      TankHeightUB = None, TankAreaUB = None, OrificeAreasUB = None, InitialLevels = None):
    DemandInput = pd.concat([DemandArray, pd.Series([0])]).reset_index(drop=True)
    CostInput = pd.concat([Cost, pd.Series([0])]).reset_index(drop=True)
    
    # ----- Set default if no parameter dictionaries are provided. 

    if OrificeAreasUB is None:
        OrificeAreasUB = {'SH': np.pi * (0.15 ** 2), 'SD': np.pi * (0.05 ** 2) , 'DO': np.pi*(0.175**2), \
                    'SD2': np.pi * (0.4 ** 2), 'DO2': np.pi * (0.125 ** 2), 'HT': np.inf, 'TD': np.inf}
    if InitialLevels is None:
        InitialLevels = {'S': 0.0, 'D': 0.0, 'H': 0.0, 'T': 0.0}
    
    dct = {None: {
        # ----- SETS
        't': {None: list(range(289))}, # This can be changed 
        'S': {None: list(range(n_scenarios))},
        'Tanks': {None: ['S', 'H', 'D', 'T']},
        'Outlets' : {None: ['SH', 'SD', 'SD2', 'DO', 'HT', 'TD', 'DO2']},
        'CondOutlets' : {None: ['SH', 'SD2', 'DO2']}, 
        'p': {None: list(range(int(len(A)/4)))},
        'q': {None: list(range(int(len(H)/4)))},
        'n': {None: list(range(int(len(OH)/3)))},
        'r' : {None: [1, 2, 3]},

         # ----- PARAMETERS
        'a' : A,
        'h' : H, 
        # ----- Rainfall 
        'RainIn': RainArray,
        'Demand': dict(enumerate(DemandInput)),
        'FreshwaterCost': dict(enumerate(CostInput)),
        
        
        'OrificeArea': OrificeAreasUB, 
        'OH':  OH,
        'InitialLevel': InitialLevels,
            }}
    
    return dct

def CompileRainfall(RFSet):
    dct = dict()
    count = 0
    for array in RFSet: 
        rf = pd.concat([array, pd.Series([0])]).reset_index(drop = True)
        dct[count] = rf
        count += 1
        
    df = pd.DataFrame(dct)
    df = df.unstack()
    df = df.swaplevel()
    dct = dict(df)
    return dct

#%% 

def SolveInstance(inst, solver = 'cplex', PrintSolverOutput = True, Summarise = True, Gap = None, TimeLimit = 10800):
    """
    
    """
    opt = pyo.SolverFactory(solver)
    
    if Gap is None: 
        Gap = 1e-4
    opt.options['mipgap'] = Gap
    
    #opt.options['feasopt'] == True
    opt.options['timelimit'] = TimeLimit
    
    opt_results = opt.solve(inst, tee = PrintSolverOutput, )
    
    
    if Summarise == True: 
        # ----- Reshape and summarize results
        Opt_Level = pd.DataFrame.from_dict(inst.Level.extract_values(), orient = 'index', columns = ['Level'])
        Opt_Level.index = pd.MultiIndex.from_tuples(Opt_Level.index, names = ['Tank', 'Timestep'])
        Opt_Level = Opt_Level.reset_index().pivot(index = 'Timestep', columns = 'Tank', values = 'Level')
        
        Opt_Discharge = pd.DataFrame.from_dict(inst.Discharge.extract_values(), orient = 'index', columns = ['Discharge'])
        Opt_Discharge.index = pd.MultiIndex.from_tuples(Opt_Discharge.index, names = ['Outlet', 'Timestep'])
        Opt_Discharge = Opt_Discharge.reset_index().pivot(index = 'Timestep', columns = 'Outlet', values = 'Discharge')
        
        Rainin = pd.DataFrame.from_dict(inst.RainIn.extract_values(), orient = 'index', columns = ['RainIn'])
        
        Opt_OF = pd.DataFrame.from_dict(inst.Overflow.extract_values(), orient = 'index', columns = ['Overflow'])
        
        FW = pd.DataFrame.from_dict(inst.Freshwater.extract_values(), orient = 'index', columns = ['Freshwater'])
       
        results = pd.concat([Rainin, Opt_Level, Opt_Discharge, Opt_OF, FW], axis = 1)
        
        # ----- 
        # TODO: Tank Parameters
        inst.TankHeight.pprint()
        inst.TankArea.pprint()
        #inst.OrificeHeight.pprint()
        #inst.OrificeArea.pprint()
        #inst.HTOnPt.pprint()
        #inst.HTOffPt.pprint()
        # Heights = inst.TankHeight.extract_values()
        # inst.TankArea.extract_values()
        # inst.OrificeHeight.extract_values()
        # inst.OrificeArea.extract_values()
        return {'Results':results, 'SolverStatus': opt_results, 'Instance': inst, } 
    
    else:
        return opt_results