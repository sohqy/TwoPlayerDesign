"""
RR Model.

Created 12 July 2022

"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns

# ========== EQUATION DEFINITIONS
def ObjFx(m):
    return sum(m.Overflow['S', t] for t in m.T)

# ----- Flow definitions
def Inflow(m, j, t):
    if j == 'S':
        return m.RainIn[t] + sum(m.Overflow[j, t] for j in m.OFSTanks)
    elif j == 'D':
        return m.TotalDischarge['S', t] - m.AdjDischarge['SH', t]
    elif j == 'H':
        return m.AdjDischarge['SH', t] + m.Overflow['T', t]
    elif j == 'T':
        return m.TotalDischarge['H', t]

def Outflow(m, j, t):
    """ Used to calculate theoretical rates of output. This is then adjusted after. """
    if j == 'S':
        return m.Discharge['SD', t] + m.Discharge['SD2', t] + m.Discharge ['SH', t]
    elif j == 'D':
        return m.Discharge['DO', t] + m.Discharge['DO2', t]
    elif j == 'H':
        return m.Discharge['HT', t]
    elif j == 'T':
        return m.Discharge['TD', t]


# ----- Tank mass balances
def MassBalancesFX(m, j, t):
    if t == m.t[-1]:
        return pyo.Constraint.Skip

    else:
        return m.Overflow[j, t+1] / m.TankArea[j] + m.Level[j, t+1] == m.Level[j, t] + ( \
                m.alpha[j] * Inflow(m, j, t) + (1 - m.alpha[j]) * Inflow(m, j, t+1) \
                - m.beta[j] * m.TotalDischarge[j, t] - (1 - m.beta[j]) * m.TotalDischarge[j, t+1] \
                ) / m.TankArea[j]


# ----- Freshwater and Demand satisfaction
def FreshwaterFx(m, t):
    return m.Demand[t] == m.Freshwater[t] + m.Discharge['TD', t]

def FreshwaterCostFx(m, t):
    return m.CostF[t] == m.Freshwater[t] * m.FreshwaterCost[t]


# ----- Overflow
def OverflowX1(m, j, t):
    """ Overflow equation definition. X2 = 0. """
    return m.TankArea[j] * m.Level[j, t] - m.TankArea[j] * m.TankHeight[j]\
            + m.alpha[j] * Inflow(m, j, t) + (1 - m.alpha[j]) * Inflow(m, j, t+1) \
            - m.beta[j] * m.TotalDischarge[j, t] - (1 - m.beta[j]) * m.TotalDischarge[j, t+1]

def OverflowFx1(m, j, t):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[j, t+1] >= OverflowX1(m, j, t)

def OverflowFx2(m, j, t):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[j, t+1] <= OverflowX1(m, j, t) + m.BigM * (1 - m.OverflowBinary[j, t+1])

def OverflowFx3(m, j, t):
    if t == m.t[-1]:
        return pyo.Constraint.Skip

    else:
        if j == 'S':
            Upperbound = m.Epsilon
        else:
            Upperbound = m.BigM

        return m.Overflow[j, t+1] <= Upperbound * m.OverflowBinary[j, t+1]


# ----- Conditional Flows
def CondFlowX1(m, c, t):
    if c in ['SD2', 'SH']:
        source = 'S'
    elif c == 'DO2':
        source = 'D'

    return 2.66 * m.OrificeArea[c] * (m.Level[source, t] - m.OrificeHeight[c]) * m.DeltaT

def CondFlowFx1(m, c, t):
    return m.Discharge[c, t] >= CondFlowX1(m, c, t)

def CondFlowFx2(m, c, t):
    return m.Discharge[c, t] <= CondFlowX1(m, c, t) + m.BigM * (1 - m.CondBinary[c, t])

def CondFlowFx3(m, c, t):
    return m.Discharge[c, t] <= m.BigM * m.CondBinary[c, t]

# ----- Unconditional Flows
def DOFlowFx(m, t):
    """ Inflexible, unconditional flow out of the system. """
    return m.Discharge['DO', t] == 2.66 * m.OrificeArea['DO'] * m.Level['D', t] * m.DeltaT

def SDFlowFx(m, t):
    """ Inflexible, unconditional flow to the detention tank. """
    return m.Discharge['SD', t] == 2.66 * m.OrificeArea['SD'] * m.Level['S', t] * m.DeltaT

# ----- Total Discharge
def TotalDischargeX1(m, j, t):
    return m.Level[j, t] * m.TankArea[j] + Inflow(m, j, t)

def TotalDischargeFx1(m, j, t):
    return m.TotalDischarge[j, t] <= TotalDischargeX1(m, j, t)

def TotalDischargeFx2(m, j, t):
    return m.TotalDischarge[j, t] <= Outflow(m, j, t)

def TotalDischargeFx3(m, j, t):
    return m.TotalDischarge[j, t] >= TotalDischargeX1(m, j, t) - m.BigM * (1 - m.DischargeBinary[j, t])

def TotalDischargeFx4(m, j, t):
    return m.TotalDischarge[j, t] >= Outflow(m, j, t) - m.BigM * m.DischargeBinary[j, t]

# ----- SH rate adjustment
def RelativeSHVariable(m):
    return m.RelSHScalar == m.OrificeArea['SH'] / (m.OrificeArea['SD'] + m.OrificeArea['SD2'])

def AdjustSHRateFx1(m, t):
    return m.AdjDischarge['SH', t] <= m.DischargeLimit['SH'] * m.CondBinary['SH', t]

def AdjustSHRateFx2(m, t):
    return m.AdjDischarge['SH', t] <= m.RelSHScalar * m.TotalDischarge['S', t]

def AdjustSHRateFx3(m, t):
    return m.AdjDischarge['SH', t] >= m.RelSHScalar * m.TotalDischarge['S', t] - (1 - m.CondBinary['SH', t]) * m.BigM


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

# ----- Vol Conservation
def VolumeConservation(m,):
    return sum(m.RainIn[t] - m.TotalDischarge['D', t] - m.TotalDischarge['T', t] for t in m.t) + sum(m.Level[j, 0] * m.TankArea[j] for j in m.Tanks) \
            == sum(m.Level[j, m.t[-1]] * m.TankArea[j] for j in m.Tanks) + (1 - m.alpha['S']) * m.Overflow['S', m.t[-1]]

# ----- Bounds and initialization
def LevelInitFx(m, j):
    return m.Level[j, 0] == m.InitialLevel[j]

def OverflowInitFx(m, j):
    return m.Overflow[j, 0] == 0.0

def DischargeInitFx(m, O):
    """ Initializes Discharge variables """
    return m.Discharge[O, 0] == 0.0

def RelayInitFx(m, j):
    return m.RelayBinary[3, 0] == 1

def LevelBounds(m, j, t):
    return m.Level[j, t] <= m.TankHeight[j]

def DischargeBounds(m, O, t):
    return m.Discharge[O, t] <= m.DischargeLimit[O]

def CalcDischargeLim(m, O):
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


# ----- Rainfall Behavioral Constraints
def RainRampUpRate(m, t):
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.RainIn[t] <= m.RainIn[t-1] + m.UpperR * m.DeltaT

def RainRampDownRate(m, t):
    if t == 0:
        return pyo.Constraint.Skip
    else:
        return m.RainIn[t] >= m.RainIn[t-1] + m.LowerR * m.DeltaT

def HorizonLimit(m):
    return sum(m.RainIn[t] for t in m.t) <= m.RainHorizonMax

def WindowLimit(m, t):
    """ Maximum rainfall volume within the stipulated window size. """
    WindowMin = int(t - m.WindowSize)       # window lowest index. If index is smaller than window size, skip.
    if WindowMin <= m.WindowSize: # Only start generating constraints for when we can have full windows.
        return pyo.Constraint.Skip
    else:
        Window = list(range(WindowMin, t+1))
        return sum(m.RainIn[w] for w in Window) <= m.RainWindowMax

def SetRainFinal(m, t):
    return m.RainIn[m.t[-1]] == 0.0

def RainInBounds(m, t):
    return m.RainIn[t] <= m.RainTSMax

# ========== FUNCTION WRAPPERS
def CreateAbstractModel(DeltaT = 300, CatchmentSize = 15.7, Eps = 5000):
    m = pyo.AbstractModel()

    # ----- SETS
    m.t = pyo.Set(ordered = True, doc = 'Discretized simulation timesteps')
    m.T = pyo.Set(ordered = True, doc = 'Obj FX included timesteps [1, ..., 288]')
    m.Tanks = pyo.Set(doc = 'Tanks within system')
    m.OFSTanks = pyo.Set(doc = 'Tanks with overflow into the S tank')
    m.Outlets = pyo.Set(doc = 'Openings within the system')
    m.CondOutlets = pyo.Set(doc = 'Conditional flow outlets')

    m.r = pyo.Set(initialize = [1, 2, 3], doc = 'Pump operational regions')

    # ----- PARAMETERS
    m.TankArea = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    m.TankHeight = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    m.InitialLevel = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    m.alpha = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    m.beta = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)

    m.OrificeArea = pyo.Param(m.Outlets, within = pyo.NonNegativeReals)
    m.OrificeHeight = pyo.Param(m.Outlets, within = pyo.NonNegativeReals)

    m.FreshwaterCost = pyo.Param(m.t, within = pyo.NonNegativeReals)
    m.Demand = pyo.Param(m.t, within = pyo.NonNegativeReals)

    # ----- SCALARS
    m.DeltaT = DeltaT
    m.Epsilon = Eps
    m.BigM = Eps
    m.RelSHScalar = pyo.Var(within = pyo.NonNegativeReals)

    m.HTOnPt = 1.0
    m.HTOffPt = 0.3

    m.UpperR = 11.4/DeltaT * CatchmentSize
    m.LowerR = -8.8/DeltaT * CatchmentSize
    m.WindowSize = 12.0 * 2
    m.RainHorizonMax = 181.2 * CatchmentSize
    m.RainWindowMax = 100 * CatchmentSize
    m.RainTSMax = 20 * CatchmentSize

    # ----- VARIABLES
    # Binary variables
    m.CondBinary = pyo.Var(m.CondOutlets, m.t, within = pyo.Binary)
    m.OverflowBinary = pyo.Var(m.Tanks, m.t, within = pyo.Binary)
    m.RelayBinary = pyo.Var(m.r, m.t, within = pyo.Binary)
    m.DischargeBinary = pyo.Var(m.Tanks, m.t, within = pyo.Binary)

    # Continuous variables
    m.Level = pyo.Var(m.Tanks, m.t, within = pyo.NonNegativeReals)
    m.Overflow = pyo.Var(m.Tanks, m.t, within = pyo.NonNegativeReals)
    m.Discharge = pyo.Var(m.Outlets, m.t, within = pyo.NonNegativeReals)
    m.TotalDischarge = pyo.Var(m.Tanks, m.t, within = pyo.NonNegativeReals)
    m.AdjDischarge = pyo.Var(m.CondOutlets, m.t, within = pyo.NonNegativeReals)

    m.Freshwater = pyo.Var(m.t, within = pyo.NonNegativeReals)
    m.CostF = pyo.Var(m.t, within = pyo.NonNegativeReals)
    m.DischargeLimit = pyo.Var(m.Outlets, within = pyo.NonNegativeReals)

    # Decision variable
    m.RainIn = pyo.Var(m.t, within = pyo.NonNegativeReals)

    # ----- EQUATIONS
    m.Obj = pyo.Objective(rule = ObjFx, sense = pyo.maximize)

    # System Dynamics
    m.MassBalances = pyo.Constraint(m.Tanks, m.t, rule = MassBalancesFX)
    m.FreshwaterUse = pyo.Constraint(m.t, rule = FreshwaterFx)
    m.Cost = pyo.Constraint(m.t, rule = FreshwaterCostFx)

    m.Overflow1 = pyo.Constraint(m.Tanks, m.t, rule = OverflowFx1)
    m.Overflow2 = pyo.Constraint(m.Tanks, m.t, rule = OverflowFx2)
    m.Overflow3 = pyo.Constraint(m.Tanks, m.t, rule = OverflowFx3)

    m.CondFlow1 = pyo.Constraint(m.CondOutlets, m.t, rule = CondFlowFx1)
    m.CondFlow2 = pyo.Constraint(m.CondOutlets, m.t, rule = CondFlowFx2)
    m.CondFlow3 = pyo.Constraint(m.CondOutlets, m.t, rule = CondFlowFx3)

    m.DOFlow = pyo.Constraint(m.t, rule = DOFlowFx)
    m.SDFlow = pyo.Constraint(m.t, rule = SDFlowFx)

    m.TotalDischarge1 = pyo.Constraint(m.Tanks, m.t, rule = TotalDischargeFx1)
    m.TotalDischarge2 = pyo.Constraint(m.Tanks, m.t, rule = TotalDischargeFx2)
    m.TotalDischarge3 = pyo.Constraint(m.Tanks, m.t, rule = TotalDischargeFx3)
    m.TotalDischarge4 = pyo.Constraint(m.Tanks, m.t, rule = TotalDischargeFx4)

    m.RelVarDef = pyo.Constraint(rule = RelativeSHVariable)
    m.AdjSHRate1 = pyo.Constraint(m.t, rule = AdjustSHRateFx1)
    m.AdjSHRate2 = pyo.Constraint(m.t, rule = AdjustSHRateFx2)
    m.AdjSHRate3 = pyo.Constraint(m.t, rule = AdjustSHRateFx3)

    m.RB = pyo.Constraint(m.t, rule = RelayBinariesFX)
    m.HT1 = pyo.Constraint(m.t, rule = HTRelay1)
    m.HT2 = pyo.Constraint(m.t, rule = HTRelay2)
    m.HT3 = pyo.Constraint(m.t, rule = HTRelay3)
    m.HT4 = pyo.Constraint(m.t, rule = HTRelay4)
    m.HT5 = pyo.Constraint(m.t, rule = HTRelay5)
    m.HT6 = pyo.Constraint(m.t, rule = HTRelay6)

    m.VolCons = pyo.Constraint(rule = VolumeConservation)

    m.InitLevel = pyo.Constraint(m.Tanks, rule = LevelInitFx)
    m.InitOF = pyo.Constraint(m.Tanks, rule = OverflowInitFx)
    m.InitiDischarge = pyo.Constraint(m.Outlets, rule = DischargeInitFx)
    m.InitRelay = pyo.Constraint(rule = RelayInitFx)
    m.BoundLevel = pyo.Constraint(m.Tanks, m.t, rule = LevelBounds)
    m.BoundDischarge = pyo.Constraint(m.Outlets, m.t, rule = DischargeBounds)

    m.CalcLimits = pyo.Constraint(m.Outlets, rule = CalcDischargeLim)

    # Rainfall behaviors
    m.RFIncrement = pyo.Constraint(m.t, rule = RainRampUpRate)
    m.RFDecrement = pyo.Constraint(m.t, rule = RainRampDownRate)
    m.HorizonTotal = pyo.Constraint(rule = HorizonLimit)
    m.WindowTotal = pyo.Constraint(m.t, rule = WindowLimit)
    m.BoundRain = pyo.Constraint(m.t, rule = RainInBounds)
    m.FinalRain = pyo.Constraint(m.t, rule = SetRainFinal)

    return m

def CompileModelParams(Parameters, DemandArray, Cost = None, InitialLevels = None, alpha = None, beta = None, ):
    # ----- Unpack Parameters
    OrificeAreas = Parameters['OrificeAreas']
    OrificeHeights = Parameters['OrificeHeights']
    TankHeights = Parameters['TankHeights']
    TankAreas = Parameters['TankAreas']

    # -----
    DemandInput = pd.concat([DemandArray, pd.Series([0])]).reset_index(drop=True)
    n_timesteps = len(DemandInput)
    if Cost is None:
        CostInput = np.ones(n_timesteps)
    else:
        CostInput = pd.concat([Cost, pd.Series([0])]).reset_index(drop=True)

    # ----- Set initial levels
    if InitialLevels is None:
        InitialLevels = {'S': 0.0, 'D': 0.0, 'H': 0.0, 'T': 0.0}

    # -----
    if alpha is None:
        alpha = {'S': 1.0, 'D': 1.0, 'H': 1.0, 'T': 1.0}

    if beta is None:
        beta = {'S': 1.0, 'D': 1.0, 'H': 1.0, 'T': 1.0}

    # -----  data
    dct = {None: {
        't': {None: list(range(n_timesteps))}, # This can be changed
        'T': {None: list(range(1, n_timesteps-80))},
        'Tanks': {None: ['S', 'H', 'D', 'T']},
        'OFSTanks': {None: ['S', 'H', 'D']},
        'Outlets' : {None: ['SH', 'SD', 'SD2', 'DO', 'HT', 'TD', 'DO2']},
        'CondOutlets' : {None: ['SH', 'SD2', 'DO2']},
        'r' : {None: [1, 2, 3]},
        'alpha' : alpha,
        'beta': beta,

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

def SolveInstance(instance, solver = 'cplex', PrintSolverOutput = True,\
                        Gap = None, TimeLimit = 300):
    """ Solves an instance. """
    opt = pyo.SolverFactory(solver)

    # ----- Optimisation options
    if Gap is None:
        Gap = 1e-4
    opt.options['mipgap'] = Gap
    opt.options['timelimit'] = TimeLimit

    opt_results = opt.solve(instance, tee = PrintSolverOutput)
    #ObjVal = pyo.value(instance.Obj)

    #return {'Instance': instance, 'ObjValue': ObjVal, 'SolverOutput': opt_results}
    return {'Instance': instance, 'SolverOutput': opt_results}


#%% ===== TEST FUNCTIONS
def TestExistingRF(m, data, RF):

    def fx(m):
        return sum(m.RainIn[t] for t in m.T)

    inst = m.create_instance(data)
    for t in range(len(RF)):
        inst.RainIn[t].fix(RF[t])

    inst.Obj.deactivate()
    inst.ObjFx = pyo.Objective(rule = fx, sense = pyo.maximize)

    # deactivate RF constraints
    inst.RFIncrement.deactivate()
    inst.RFDecrement.deactivate()
    inst.HorizonTotal.deactivate()
    inst.WindowTotal.deactivate()
    inst.BoundRain.deactivate()
    inst.FinalRain.deactivate()

    return inst


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
    Opt_OF.index = pd.MultiIndex.from_tuples(Opt_OF.index, names = ['Tank', 'Timestep'])
    Opt_OF = Opt_OF.reset_index().pivot(index = 'Timestep', columns = 'Tank', values = 'Overflow')
    Opt_OF.rename(columns = {'S' : 'OF S', 'D' : 'OF D', 'H':'OF H', 'T':'OF T'}, inplace = True)

    Total_Discharge = pd.DataFrame.from_dict(inst.TotalDischarge.extract_values(), orient = 'index', columns = ['Total Discharge'])
    Total_Discharge.index = pd.MultiIndex.from_tuples(Total_Discharge.index, names = ['Tank', 'Timestep'])
    Total_Discharge = Total_Discharge.reset_index().pivot(index = 'Timestep', columns = 'Tank', values = 'Total Discharge')
    Total_Discharge.rename(columns = {'S' : 'Discharge S', 'D' : 'Discharge D', 'H':'Discharge H', 'T':'Discharge T'}, inplace = True)

    results = pd.concat([Rainin, Opt_Level, Opt_Discharge, Opt_OF,  Total_Discharge], axis = 1)

    return results

def PlotDynamics(Dataframe, instance):
    plt.figure()
    colors = sns.color_palette()

    plt.subplot(311)
    plt.plot(Dataframe['RainIn'], label = 'RuinousRain')
    plt.plot(Dataframe['OF S'], label = 'Overflow S')
    plt.plot(Dataframe['OF D'], label = 'Overflow D')
    plt.plot(Dataframe['OF H'], label = 'Overflow H')
    plt.plot(Dataframe['OF T'], label = 'Overflow T')
    plt.legend()

    # ----- Plot Tank Dynamics
    cnt = 0
    plt.subplot(312)
    plt.plot(Dataframe.H, label = 'Harvested Volume', color = colors[cnt])
    plt.axhline(instance.TankHeight.extract_values()['H'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    cnt += 1
    plt.plot(Dataframe.S, label = 'Separation Tank Levels', color = colors[cnt])
    plt.axhline(instance.TankHeight.extract_values()['S'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    plt.axhline(instance.OrificeHeight.extract_values()['SH'], xmin = 0, xmax = 1, color = 'black', linestyle = '-', alpha = 0.5)
    plt.axhline(instance.OrificeHeight.extract_values()['SD2'], xmin = 0, xmax = 1, color = 'black', linestyle = '-', alpha = 0.5)
    cnt +=1
    plt.plot(Dataframe.D, label = 'Detention Tank Levels', color = colors[cnt])
    plt.axhline(instance.TankHeight.extract_values()['D'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    plt.axhline(instance.OrificeHeight.extract_values()['DO2'], xmin = 0, xmax = 1, color = 'black', linestyle = '-', alpha = 0.5)
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
