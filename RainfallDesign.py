# -*- coding: utf-8 -*-
"""
RAINFALL 'OPTIMISATION' MODEL (PLAYER 2)
========================================

Created on Fri Apr  1 17:19:03 2022

@author: sohqi
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd 

def ObjFx(m):
    """ Maximise overflow over entire simulation horizon. """ 
    return sum(m.Overflow[t] for t in m.t) # - yield

# ----- Tank equations 
def SepBalanceFx(m, t):
    """ Discretized mass balance for the Separation Tank. """
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else: 
        return m.Overflow[t+1]/m.TankArea['S'] + m.Level['S', t+1] == m.Level['S', t] \
                + (m.RainIn[t] + m.Overflow[t] - m.Discharge['SH', t] - m.Discharge['SD', t] - m.Discharge['SD2', t]) / m.TankArea['S']
        
def DetBalanceFx(m, t):
    """ Discretized mass balance for the Detention Tank. """
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Level['D', t+1] == m.Level['D', t] \
            + (m.Discharge['SD', t] + m.Discharge['SD2', t] - m.Discharge['DO', t] - m.Discharge['DO2', t]) / m.TankArea['D']
            
def HarBalanceFx(m, t):
    """ Discretized mass balance for the Harvesting Tank. """
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else: 
        return m.Level['H', t+1] == m.Level['H', t] \
            + (m.Discharge['SH', t] - m.Discharge['HT', t]) / m.TankArea['H'] 
            
def TrtBalanceFx(m, t):
    """ Discretized mass balance for the Treatment Tank. """
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Level['T', t+1] == m.Level['T', t] \
            + (m.Discharge['HT', t] - m.Discharge['TD', t]) / m.TankArea['T']

# ----- Demand and Freshwater use
def FreshwaterFx(m, t):
    """ Freshwater top up for demand. """
    return m.Demand[t] == m.Freshwater[t] + m.Discharge['TD', t]             

def FreshwaterCostFx(m, t):
    """ Cost of using freshwater to top up for demand. """
    return m.CostF[t] == m.Freshwater[t] * m.FreshwaterCost[t]

# ----- 
def OverflowFx(m, t):
    """ Definition of system overflow. """
    # if t == m.t[-1]:
    #     return pyo.Constraint.Skip
    # else:
    #     return m.Overflow[t+1] >= m.TankArea['S'] * m.Level['S', t] + m.RainIn[t] \
    #         + m.Overflow[t] - m.Discharge['SH', t] - m.Discharge['SD', t] - m.Discharge['SD2', t] \
    #             - m.TankArea['S'] * m.TankHeight['S']
    
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t] >= m.TankArea['S'] * m.Level['S', t] + m.RainIn[t] \
            + m.Overflow[t-1] - m.Discharge['SH', t] - m.Discharge['SD', t] - m.Discharge['SD2', t] \
                - m.TankArea['S'] * m.TankHeight['S']

def OverflowFxUB2(m, t):
    """ Upper bound effective for setting value when condition is True. """
    # if t == m.t[-1]:
    #     return pyo.Constraint.Skip
    # else:
    #     return m.Overflow[t+1] <= m.TankArea['S'] * m.Level['S', t] + m.RainIn[t] \
    #         + m.Overflow[t] - m.Discharge['SH', t] - m.Discharge['SD', t] - m.Discharge['SD2', t] \
    #             - m.TankArea['S'] * m.TankHeight['S'] \
    #                 - (m.DischargeLimit['SH'] + m.DischargeLimit['SD'] + m.DischargeLimit['SD2'] +  m.TankArea['S'] * m.TankHeight['S']) * m.OverflowBinary[t] \
    #                     + (m.DischargeLimit['SH'] + m.DischargeLimit['SD'] + m.DischargeLimit['SD2'] +  m.TankArea['S'] * m.TankHeight['S']) 
                
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t] <= m.TankArea['S'] * m.Level['S', t] + m.RainIn[t] \
            + m.Overflow[t-1] - m.Discharge['SH', t] - m.Discharge['SD', t] - m.Discharge['SD2', t] \
                - m.TankArea['S'] * m.TankHeight['S'] \
                    - (m.DischargeLimit['SH'] + m.DischargeLimit['SD'] + m.DischargeLimit['SD2'] +  m.TankArea['S'] * m.TankHeight['S']) * m.OverflowBinary[t] \
                        + (m.DischargeLimit['SH'] + m.DischargeLimit['SD'] + m.DischargeLimit['SD2'] +  m.TankArea['S'] * m.TankHeight['S']) 
                
def OverflowLm(m, t):
    """ Desired upper bound for overflow. """
    return m.Overflow[t] <= m.Epsilon * m.OverflowBinary[t]

# ----- Conditional Flows

def SHLowerFx(m, t):
    """ Lower bound effective when condition is True. Deactivated for increased flexibility. """
    return m.Discharge['SH', t] >= 2.66 * m.OrificeArea['SH'] * (m.Level['S', t] - m.OrificeHeight['SH']) * m.DeltaT

def SHUpperFx1(m, t):
    """ Sets upper bound of 0 for SH flow when conditions are not met """
    return m.Discharge['SH', t] <= 2.66 * m.OrificeArea['SH'] * m.DeltaT * m.CondBinary['SH', t] \
        * (m.TankHeight['S'] - m.OrificeHeight['SH'])

def SHUpperFx2(m, t):
    """ Sets upper bound when condition is True. """
    return m.Discharge['SH', t] <= 2.66 * m.OrificeArea['SH'] * m.DeltaT * (m.Level['S', t] - m.OrificeHeight['SH']) \
        - 2.66 * m.OrificeArea['SH'] * m.DeltaT * m.CondBinary['SH', t] * m.OrificeHeight['SH'] \
            + 2.66 * m.OrificeArea['SH'] * m.DeltaT * m.OrificeHeight['SH']
    
# -----             
def SD2LowerFx(m, t):
    """ Lower bound for SD2, effective when condition is True."""
    return m.Discharge['SD2', t] >= 2.66 * m.OrificeArea['SD2'] * (m.Level['S', t] - m.OrificeHeight['SD2']) * m.DeltaT

def SD2UpperFx1(m, t):
    """ Sets upper bound of 0 for SD2 flow when conditions are not met """
    return m.Discharge['SD2', t] <= 2.66 * m.OrificeArea['SD2'] * m.DeltaT * m.CondBinary['SD2', t] \
        * (m.TankHeight['S'] - m.OrificeHeight['SD2'])
        
def SD2UpperFx2(m, t): 
    """ Sets upper bound when condition is True. """
    return m.Discharge['SD2', t] <= 2.66 * m.OrificeArea['SD2'] * m.DeltaT * (m.Level['S', t] - m.OrificeHeight['SD2']) \
        - 2.66 * m.OrificeArea['SD2'] * m.DeltaT * m.CondBinary['SD2', t] * m.OrificeHeight['SD2'] \
            + 2.66 * m.OrificeArea['SD2'] * m.DeltaT * m.OrificeHeight['SD2']
            
# ----- 
def DO2LowerFx(m, t):
    """ Lower bound for DO2, effective when flow is True """
    return m.Discharge['DO2', t] >= 2.66 * m.OrificeArea['DO2'] * (m.Level['D', t] - m.OrificeHeight['DO2']) * m.DeltaT

def DO2UpperFx1(m, t):
    """ Sets upper bound of 0 for SD2 flow when conditions are not met """
    return m.Discharge['DO2', t] <= 2.66 * m.OrificeArea['DO2'] * m.DeltaT * m.CondBinary['DO2', t] \
        * (m.TankHeight['D'] - m.OrificeHeight['DO2'])
        
def DO2UpperFx2(m, t): 
    """ Sets upper bound when condition is True. """
    return m.Discharge['DO2', t] <= 2.66 * m.OrificeArea['DO2'] * m.DeltaT * (m.Level['D', t] - m.OrificeHeight['DO2']) \
        - 2.66 * m.OrificeArea['DO2'] * m.DeltaT * m.CondBinary['DO2', t] * m.OrificeHeight['DO2'] \
            + 2.66 * m.OrificeArea['DO2'] * m.DeltaT * m.OrificeHeight['DO2']

# ----- Unconditional Flows             
def DOFlowFx(m, t):
    """ Inflexible, unconditional flow out of the system. """
    return m.Discharge['DO', t] == 2.66 * m.OrificeArea['DO'] * m.Level['D', t] * m.DeltaT

def SDFlowFx(m, t):
    """ Inflexible, unconditional flow to the detention tank. """
    return m.Discharge['SD', t] == 2.66 * m.OrificeArea['SD'] * m.Level['S', t] * m.DeltaT

# ----- Treatment pump operation 
def RelayBinariesFX(m, t):
    """ Binary selector for region of operation of pump. """
    return m.RelayBinary[1, t] + m.RelayBinary[2, t] + m.RelayBinary[3, t] == 1
    
def HTRelay1(m,t): 
    """ """
    return m.Discharge['HT', t] >= m.DischargeLimit['HT'] * m.RelayBinary[1, t]

def HTRelay2(m, t):
    """ """
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.Discharge['HT', t] <= m.Discharge['HT', t-1] - m.DischargeLimit['HT'] * m.RelayBinary[2, t] + m.DischargeLimit['HT']

def HTRelay3(m, t):
    """ """
    return m.Discharge['HT', t] <= (1 - m.RelayBinary[3, t]) * m.DischargeLimit['HT']

def HTRelay4(m, t):
    """ """
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.Discharge['HT', t] >= m.Discharge['HT', t-1] + m.DischargeLimit['HT'] * m.RelayBinary[2, t] - m.DischargeLimit['HT']

def HTRelay5(m, t):
    """ """
    return m.Level['H', t] >= m.HTOnPt * m.RelayBinary[1, t] + m.HTOffPt * m.RelayBinary[2, t]

def HTRelay6(m, t):
    """ """
    return m.Level['H', t] <= m.TankHeight['H'] * m.RelayBinary[1, t] \
        + m.HTOnPt * m.RelayBinary[2, t] + m.HTOffPt * m.RelayBinary[3, t]

# ----- Vol Cons
def VolConsFx(m, t):
    """ Overall volume conservation of the system. """
    return m.Overflow[t] + m.TankArea['S'] * m.Level['S', t] \
    + m.TankArea['H'] * m.Level['H', t] + m.TankArea['D'] * m.Level['D', t] \
        == m.TankArea['S'] * m.Level['S', 0] + m.TankArea['H'] * m.Level['H', 0] \
            + m.TankArea['D'] * m.Level['D', 0] \
                + sum(m.RainIn[k] for k in range(0, t)) - sum(m.Discharge['DO', k] for k in range(0, t)) \
                    - sum(m.Discharge['HT', k] for k in range(0, t)) - sum(m.Discharge['DO2', k] for k in range(0, t))

# ----- Variable Initialisation
def LevelInitFx(m, T):
    """ Initializes Level variables """
    return m.Level[T, 0] == m.InitialLevel[T]

def DischargeInitFx(m, O):
    """ Initializes Discharge variables """
    return m.Discharge[O, 0] == 0.0 

def OverflowInitFx(m):
    """ Initializes Overflow variable """
    return m.Overflow[0] == 0.0

def InitRelayFX(m): 
    return m.RelayBinary[3, 0] == 1

# ----- Variable Bound Assignment   
def LevelBounds(m, T, t): 
    """ Sets upper bounds for Level variables """
    return m.Level[T, t] <= m.TankHeight[T]

def DischargeBounds(m, O, t):
    """ Sets upper bounds for Discharge variables """
    return m.Discharge[O, t] <= m.DischargeLimit[O]


def DischargeLimitsFx(m, O):
    """ Calculate discharge limits.  """
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


# ----- RAINFALL PATTERN CONSTRAINTS 
def RainUpRampLimit(m, t):
    """ Maximum increase in rainfall volume between timesteps """
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.RainIn[t] <= m.RainIn[t-1] + m.UpperR * m.DeltaT 

def RainDownRampLimit(m, t):
    """ Minimum decrease in rainfall volume between timesteps """
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.RainIn[t] >= m.RainIn[t-1] + m.LowerR * m.DeltaT

def HorizonLimit(m):
    """ Maximum rainfall volume within entire simulation horizon"""
    return sum(m.RainIn[t] for t in m.t) <= m.RainMax

def WindowLimit(m, t):
    """ Maximum rainfall volume within the stipulated window size. """
    WindowMin = t - m.WindowSize
    if WindowMin <= m.WindowSize: # Only start calculating when we can have full windows. 
        return pyo.Constraint.Skip
    else: 
        Window = list(range(WindowMin, t+1))
        return sum(m.RainIn[w] for w in Window) <= m.WindowRainMax

def RainInit(m):
    """ Initialise rainfall volume. """
    return m.RainIn[0] == 0.0

def RainTSMax(m, t):
    """ Maximum rainfall volume in a single timestep. """
    return m.RainIn[t] <= m.RainTSMax

#%% 

def CreateAbstractModel(DeltaT = 300, CatchmentSize = 15.7):
    """ Create a model. """
    m = pyo.AbstractModel()
    
    # ===== SETS 
    m.t = pyo.Set(ordered = True, doc = 'Simulation timesteps')
    m.Tanks = pyo.Set(doc = 'Tank names')
    m.Outlets = pyo.Set(doc = 'Outlets in system')
    m.CondOutlets = pyo.Set(doc = 'Conditional flow outlets')
    
    m.r = pyo.Set(initialize = [1, 2, 3], doc = 'Relay binary')
    
    # ===== PARAMETERS 
    m.TankArea = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    m.TankHeight = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    m.InitialLevel = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    
    m.OrificeArea = pyo.Param(m.Outlets, within = pyo.NonNegativeReals)
    m.OrificeHeight = pyo.Param(m.Outlets, within = pyo.NonNegativeReals)
    
    m.FreshwaterCost = pyo.Param(m.t, within = pyo.NonNegativeReals)
    m.Demand = pyo.Param(m.t, within = pyo.NonNegativeReals)
    
    # ----- SCALARS 
    m.DeltaT = DeltaT
    m.Epsilon = 1000
    
    m.HTOnPt = 1.0
    m.HTOffPt = 0.3
    
    m.UpperR = 11.4/DeltaT * CatchmentSize
    m.LowerR = - 8.8/DeltaT * CatchmentSize
    m.WindowSize = 12.0 * 2           # 2 hours. 
    m.RainMax = 181.2 * CatchmentSize
    m.WindowRainMax = 100 * CatchmentSize
    m.RainTSMax = 20 * CatchmentSize # historical observation = 14.6
    
    # ===== VARIABLES 
    # ----- Binary variables
    m.CondBinary = pyo.Var(m.CondOutlets, m.t, within = pyo.Binary)
    m.OverflowBinary = pyo.Var(m.t, within = pyo.Binary)
    m.RelayBinary = pyo.Var(m.r, m.t, within = pyo.Binary)
    
    # ----- Continuous variables
    m.Level = pyo.Var(m.Tanks, m.t, within = pyo.NonNegativeReals)
    m.Freshwater = pyo.Var(m.t, within = pyo.NonNegativeReals)
    m.Overflow = pyo.Var(m.t, within = pyo.NonNegativeReals)
    m.CostF = pyo.Var(m.t, within = pyo.NonNegativeReals)
    
    m.Discharge = pyo.Var(m.Outlets, m.t, within = pyo.NonNegativeReals)
    m.DischargeLimit = pyo.Var(m.Outlets, within = pyo.NonNegativeReals)
        
    # ----- 
    m.RainIn = pyo.Var(m.t, within = pyo.NonNegativeReals)
    
    # ===== EQUATIONS 
    m.Obj = pyo.Objective(rule = ObjFx, sense = pyo.maximize)
    
    # ----- Mass balances
    m.SepTank = pyo.Constraint(m.t, rule = SepBalanceFx)
    m.HarTank = pyo.Constraint(m.t, rule = HarBalanceFx)
    m.DetTank = pyo.Constraint(m.t, rule = DetBalanceFx)
    m.TrtTank = pyo.Constraint(m.t, rule = TrtBalanceFx)
    
    # ----- Freshwater use 
    m.FreshwaterDef = pyo.Constraint(m.t, rule = FreshwaterFx)
    m.FreshwaterCosts = pyo.Constraint(m.t, rule = FreshwaterCostFx)
    
    # ----- Flows
    m.OF = pyo.Constraint(m.t, rule = OverflowFx)
    m.OFUB = pyo.Constraint(m.t, rule = OverflowFxUB2)
    m.OFLim = pyo.Constraint(m.t, rule = OverflowLm)
    
    m.SH1 = pyo.Constraint(m.t, rule = SHLowerFx)
    m.SH2 = pyo.Constraint(m.t, rule = SHUpperFx1)
    m.SH3 = pyo.Constraint(m.t, rule = SHUpperFx2)
    
    m.SD21 = pyo.Constraint(m.t, rule = SD2LowerFx)
    m.SD22 = pyo.Constraint(m.t, rule = SD2UpperFx1)
    m.SD23 = pyo.Constraint(m.t, rule = SD2UpperFx2)
    
    m.DO21 = pyo.Constraint(m.t, rule = DO2LowerFx)
    m.DO22 = pyo.Constraint(m.t, rule = DO2UpperFx1)
    m.DO23 = pyo.Constraint(m.t, rule = DO2UpperFx2)
    
    m.DO = pyo.Constraint(m.t, rule = DOFlowFx)
    m.SD = pyo.Constraint(m.t, rule = SDFlowFx)
    
    # ----- Pump treatment operation 
    m.RB = pyo.Constraint(m.t, rule = RelayBinariesFX)
    m.HT1 = pyo.Constraint(m.t, rule = HTRelay1)
    m.HT2 = pyo.Constraint(m.t, rule = HTRelay2)
    m.HT3 = pyo.Constraint(m.t, rule = HTRelay3)
    m.HT4 = pyo.Constraint(m.t, rule = HTRelay4)
    m.HT5 = pyo.Constraint(m.t, rule = HTRelay5)
    m.HT6 = pyo.Constraint(m.t, rule = HTRelay6)

    # ----- 
    m.VolumeConservation = pyo.Constraint(m.t, rule = VolConsFx)
    
    # ----- 
    m.LevelLim = pyo.Constraint(m.Tanks, m.t, rule = LevelBounds)
    m.DischargeLim = pyo.Constraint(m.Outlets, m.t, rule = DischargeBounds)
    
    # ----- Initialisation of variables 
    m.RelayInit = pyo.Constraint(rule = InitRelayFX)

    m.InitLevel = pyo.Constraint(m.Tanks, rule = LevelInitFx)
    m.InitDischarge = pyo.Constraint(m.Outlets, rule = DischargeInitFx)
    m.InitOverflow = pyo.Constraint(rule = OverflowInitFx)
    
    m.DischargeLimits = pyo.Constraint(m.Outlets, rule = DischargeLimitsFx)
    
    # ----- Rainfall Constraints. 
    m.RainfallIncrementLimit = pyo.Constraint(m.t, rule = RainUpRampLimit)
    m.RainfallDecrementLimit = pyo.Constraint(m.t, rule = RainDownRampLimit)
    m.HorizonTotal = pyo.Constraint(rule = HorizonLimit)
    m.WindowTotal = pyo.Constraint(m.t, rule = WindowLimit)
    m.InitRain = pyo.Constraint(rule = RainInit)
    m.TimestepMax = pyo.Constraint(m.t, rule = RainTSMax)

    return m

#%% 

def CompileModelParams(Parameters, DemandArray, Cost = None, InitialLevels = None):
    # ----- Unpack Parameters 
    OrificeAreas = Parameters['OrificeAreas']
    OrificeHeights = Parameters['OrificeHeights']
    TankHeights = Parameters['TankHeights']
    TankAreas = Parameters['TankAreas'] 
    
    # ----- 
    DemandInput = pd.concat([DemandArray, pd.Series([0])]).reset_index(drop=True)
    if Cost is None:
        CostInput = np.ones(289)
    else:
        CostInput = pd.concat([Cost, pd.Series([0])]).reset_index(drop=True)
    
    # ----- Set initial levels
    if InitialLevels is None:
        InitialLevels = {'S': 0.0, 'D': 0.0, 'H': 0.0, 'T': 0.0}
    
    # ----- Compile data 
    dct = {None: {
        't': {None: list(range(289))}, # This can be changed 
        'Tanks': {None: ['S', 'H', 'D', 'T']},
        'Outlets' : {None: ['SH', 'SD', 'SD2', 'DO', 'HT', 'TD', 'DO2']},
        'CondOutlets' : {None: ['SH', 'SD2', 'DO2']}, 
        'r' : {None: [1, 2, 3]},
        
        # ----- 
        'Demand': dict(enumerate(DemandInput)),
        'FreshwaterCost': dict(enumerate(CostInput)),
        
        # ----- 
        'OrificeArea': OrificeAreas,
        'OrificeHeight': OrificeHeights,
        'TankHeight': TankHeights,
        'TankArea': TankAreas,
        'InitialLevel': InitialLevels,
        
        }}
    
    return dct


def CreateSolveInstance(m, data, solver = 'cplex', PrintSolverOutput = True,\
                        Gap = None, TimeLimit = 300):
    """ Creates an instance and solves this based on the data provided. """
    instance = m.create_instance(data)
    opt = pyo.SolverFactory(solver)
    
    # ----- Optimisation options
    if Gap is None: 
        Gap = 1e-4
    opt.options['mipgap'] = Gap
    opt.options['timelimit'] = TimeLimit
    
    
    opt_results = opt.solve(instance, tee = PrintSolverOutput)
    ObjVal = pyo.value(instance.Obj)
    
    return {'Instance': instance, 'ObjValue': ObjVal, 'SolverOuptut': opt_results}


#%% Testing functions 
import matplotlib.pyplot as plt
import seaborn as sns

def CreateFakeDimensions():
    P = {'TankAreas': {'S': 20.0, 'D': 120.0, 'H': 10.0, 'T': 120.0}, 
         'TankHeights': {'S': 2.0, 'D': 3.0, 'H': 1.5, 'T': 1.25}, 
         'OrificeHeights': {'SH': 0.4, 'SD2': 1.0, 'DO2': 0.6, 'SD': 0.0, 'DO': 0.0}, 
         'OrificeAreas': {'SH': 0.07068583470577035, 'SD': 0.007853981633974483, 'DO': 0.0962112750161874, 'SD2': 0.5026548245743669, 'DO2': 0.04908738521234052, 'HT': np.inf, 'TD': np.inf}
         }
    
    return P

def Summarise(inst):
    Opt_Level = pd.DataFrame.from_dict(inst.Level.extract_values(), orient = 'index', columns = ['Level'])
    Opt_Level.index = pd.MultiIndex.from_tuples(Opt_Level.index, names = ['Tank', 'Timestep'])
    Opt_Level = Opt_Level.reset_index().pivot(index = 'Timestep', columns = 'Tank', values = 'Level')

    Opt_Discharge = pd.DataFrame.from_dict(inst.Discharge.extract_values(), orient = 'index', columns = ['Discharge'])
    Opt_Discharge.index = pd.MultiIndex.from_tuples(Opt_Discharge.index, names = ['Outlet', 'Timestep'])
    Opt_Discharge = Opt_Discharge.reset_index().pivot(index = 'Timestep', columns = 'Outlet', values = 'Discharge')
    
    Rainin = pd.DataFrame.from_dict(inst.RainIn.extract_values(), orient = 'index', columns = ['RainIn'])
        
    Opt_OF = pd.DataFrame.from_dict(inst.Overflow.extract_values(), orient = 'index', columns = ['Overflow'])
        
    results = pd.concat([Rainin, Opt_Level, Opt_Discharge, Opt_OF], axis = 1)
        
    return results

def PlotDynamics(Dataframe, instance):
    plt.figure()
    colors = sns.color_palette()
    
    plt.subplot(311)
    plt.plot(Dataframe['RainIn'], label = 'RuinousRain')
    plt.plot(Dataframe['Overflow'], label = 'Overflow')
    plt.legend()
    
    # ----- Plot Tank Dynamics
    cnt = 0
    plt.subplot(312)
    plt.plot(Dataframe.H, label = 'Harvested Volume', color = colors[cnt])
    plt.axhline(instance.TankHeight.extract_values()['H'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    cnt += 1 
    plt.plot(Dataframe.S, label = 'Separation Tank Levels', color = colors[cnt])
    plt.axhline(instance.TankHeight.extract_values()['S'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    plt.axhline(instance.OrificeHeight.extract_values()['SH'], xmin = 0, xmax = 1, color = 'grey', linestyle = '--', alpha = 0.5)
    plt.axhline(instance.OrificeHeight.extract_values()['SD2'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = '--', alpha = 0.5)
    cnt +=1 
    plt.plot(Dataframe.D, label = 'Detention Tank Levels', color = colors[cnt])
    plt.axhline(instance.TankHeight.extract_values()['D'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    plt.axhline(instance.OrificeHeight.extract_values()['DO2'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = '--', alpha = 0.5)
    cnt +=1 
    plt.plot(Dataframe['T'], label = 'Treatment Tank Levels', color = colors[cnt])
    plt.axhline(instance.TankHeight.extract_values()['T'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    plt.legend()
    plt.ylabel('Tank Water Levels \n [m$^3$]')
    
    # ----- Plot Rates 
    plt.subplot(313)
    cnt = 0
    plt.plot(Dataframe.SH, label = 'Harvest Rate', color = colors[cnt])
    plt.axhline(instance.DischargeLimit.extract_values()['SH'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    cnt += 1
    plt.plot(Dataframe.SD, label = 'Primary detention rate', color = colors[cnt])
    plt.axhline(instance.DischargeLimit.extract_values()['SD'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    cnt += 1
    plt.plot(Dataframe.SD2, label = 'Seconday detention rate', color = colors[cnt])
    plt.axhline(instance.DischargeLimit.extract_values()['SD2'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    cnt += 1
    plt.plot(Dataframe.DO2, label = 'Secondary Public Outflow Rate', color = colors[cnt])
    plt.axhline(instance.DischargeLimit.extract_values()['DO2'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    plt.legend()
    plt.ylabel('Discharge Volumes \n per timestep [m$^3$/s]')
    plt.xlabel('Simulation Timestep ID')