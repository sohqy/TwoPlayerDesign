"""
CODE TITLE HERE

Created
Qiao Yan Soh. qys13@ic.ac.uk
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns

# ========== EQUATION Definitions
def ObjFx(m):
    return sum(m.Overflow[t] for t in m.T)

# ----- Flow definitions
def Inflow(m, t):
    return m.RainIn[t] + m.Overflow[t]

def Outflow(m, t):
    return m.Discharge['SD', t] + m.Discharge['SH', t] + m.Discharge['SD2', t]

# ----- Tank Mass balances
def MassBalanceFX(m, t):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t+1]/m.TankArea + m.Level[t+1] == m.Level[t] + \
        (m.alpha * Inflow(m, t) + (1 - m.alpha) * Inflow(m, t+1) \
        - m.beta * m.TotalDischarge[t] - (1 - m.beta) * m.TotalDischarge[t+1]) / m.TankArea

# ----- Overflow
def OverflowX1(m, t):
    return m.TankArea * m.Level[t] - m.TankArea * m.TankHeight \
        + m.alpha * Inflow(m, t) + (1 - m.alpha) * Inflow(m, t+1) \
        - m.beta * m.TotalDischarge[t] - (1 - m.beta) * m.TotalDischarge[t+1]

def OverflowFx1(m, t):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t+1] >= OverflowX1(m, t)

def OverflowFx2(m, t):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t+1] <= OverflowX1(m, t) + m.Epsilon * (1 - m.OverflowBinary[t+1])

def OverflowFx3(m, t):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t+1] <= m.Epsilon * m.OverflowBinary[t+1]

# ----- Conditional flows
def CondFlowX1(m, c, t):
    return 2.66 * m.OrificeArea[c] * (m.Level[t] - m.OrificeHeight[c]) * m.DeltaT

def CondFlowFx1(m, c, t):
    return m.Discharge[c, t] >= CondFlowX1(m, c, t)

def CondFlowFx2(m, c, t):
    return m.Discharge[c, t] <= CondFlowX1(m, c, t) + m.BigM2 * (1 - m.CondBinary[c, t])

def CondFlowFx3(m, c, t):
    return m.Discharge[c, t] <= m.BigM2 * m.CondBinary[c, t]

# ----- Unconditional flow
def SDFlowFx(m, t):
    """ Inflexible, unconditional flow to the detention tank. """
    return m.Discharge['SD', t] == 2.66 * m.OrificeArea['SD'] * m.Level[t] * m.DeltaT

# ----- Total Discharge
def AvailVolX1(m, t):
    return m.Level[t] * m.TankArea + Inflow(m, t)

def TotalDischargeFx1(m, t):
    return m.TotalDischarge[t] <= AvailVolX1(m, t)

def TotalDischargeFx2(m, t):
    return m.TotalDischarge[t] <= Outflow(m, t)

def TotalDischargeFx3(m, t):
    return m.TotalDischarge[t] >= AvailVolX1(m, t) - m.BigM * (1 - m.DischargeBinary[t])

def TotalDischargeFx4(m, t):
    return m.TotalDischarge[t] >= Outflow(m, t) - m.BigM * m.DischargeBinary[t]

# ----- Volume conservation
def VolumeConservation(m,):
    return sum(m.RainIn[t] - m.TotalDischarge[t] for t in m.t) + m.Level[0] * m.TankArea \
        == m.Level[m.t[-1]] * m.TankArea + (1 - m.alpha) * m.Overflow[m.t[-1]]

# ----- Bounds and initialisations
def LevelInitFx(m):
    return m.Level[0] == m.InitialLevel

def OverflowInitFx(m):
    return m.Overflow[0] == 0.0

def DischargeInitFx(m, O):
    return m.Discharge[O, 0] == 0.0

def LevelBounds(m, t):
    return m.Level[t] <= m.TankHeight

def DischargeBounds(m, O, t):
    return m.Discharge[O, t] <= m.DischargeLimit[O]

def CalcDischargeLim(m, O):
    if O in ['SH', 'SD2']:
        return m.DischargeLimit[O] == 2.66 * m.OrificeArea[O] * (m.TankHeight - m.OrificeHeight[O]) * m.DeltaT
    elif O == 'SD':
        return m.DischargeLimit[O] == 2.66 * m.OrificeArea[O] * m.TankHeight * m.DeltaT

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
    """ Rainfall signal should be taken from [0, ..., 287] """
    return m.RainIn[m.t[-1]] == 0.0

def RainInBounds(m, t):
    return m.RainIn[t] <= m.RainTSMax


# ========== FUNCTIONS AND WRAPPERS
def CreateAbstractModel(DeltaT = 300, CatchmentSize = 15.7, Eps = 2000):
    m = pyo.AbstractModel()

    # ----- SETS
    m.t = pyo.Set(ordered = True, doc = 'Discretized simulation timesteps')
    m.T = pyo.Set(ordered = True, doc = 'Obj FX included timesteps [1, ..., 288]')
    m.Outlets = pyo.Set(doc = 'Openings within the system')
    m.CondOutlets = pyo.Set(doc = 'Conditional flow outlets')

    # ----- PARAMETERS
    m.TankArea = pyo.Param(within = pyo.NonNegativeReals)
    m.TankHeight = pyo.Param(within = pyo.NonNegativeReals)
    m.InitialLevel = pyo.Param(within = pyo.NonNegativeReals)
    m.alpha = pyo.Param(within = pyo.NonNegativeReals)
    m.beta = pyo.Param(within = pyo.NonNegativeReals)

    m.OrificeArea = pyo.Param(m.Outlets, within = pyo.NonNegativeReals)
    m.OrificeHeight = pyo.Param(m.Outlets, within = pyo.NonNegativeReals)

    # ----- SCALARS
    m.DeltaT = DeltaT
    m.Epsilon = Eps
    m.BigM = 5000
    m.BigM2 = 3000

    m.UpperR = 11.4/300 * DeltaT * CatchmentSize
    m.LowerR = -8.8/300 * DeltaT * CatchmentSize
    m.WindowSize = 12.0 * 2
    m.RainHorizonMax = 181.2 * CatchmentSize
    m.RainWindowMax = 100 * CatchmentSize
    m.RainTSMax = 20 * CatchmentSize

    # ----- VARIABLES
    m.CondBinary = pyo.Var(m.CondOutlets, m.t, within = pyo.Binary)
    m.OverflowBinary = pyo.Var(m.t, within = pyo.Binary)
    m.DischargeBinary = pyo.Var(m.t, within = pyo.Binary)

    m.Level = pyo.Var(m.t, within = pyo.NonNegativeReals)
    m.Overflow = pyo.Var(m.t, within = pyo.NonNegativeReals)
    m.Discharge = pyo.Var(m.Outlets, m.t, within = pyo.NonNegativeReals)
    m.TotalDischarge = pyo.Var(m.t, within = pyo.NonNegativeReals)

    m.DischargeLimit = pyo.Var(m.Outlets, within = pyo.NonNegativeReals)

    m.RainIn = pyo.Var(m.t, within = pyo.NonNegativeReals)

    # ----- EQUATIONS
    m.Obj = pyo.Objective(rule = ObjFx, sense = pyo.maximize)
    m.MassBalances = pyo.Constraint(m.t, rule = MassBalanceFX)

    m.Overflow1 = pyo.Constraint(m.t, rule = OverflowFx1)
    m.Overflow2 = pyo.Constraint(m.t, rule = OverflowFx2)
    m.Overflow3 = pyo.Constraint(m.t, rule = OverflowFx3)

    m.CondFlow1 = pyo.Constraint(m.CondOutlets, m.t, rule = CondFlowFx1)
    m.CondFlow2 = pyo.Constraint(m.CondOutlets, m.t, rule = CondFlowFx2)
    m.CondFlow3 = pyo.Constraint(m.CondOutlets, m.t, rule = CondFlowFx3)

    m.SDFlow = pyo.Constraint(m.t, rule = SDFlowFx)

    m.TotalDischarge1 = pyo.Constraint(m.t, rule = TotalDischargeFx1)
    m.TotalDischarge2 = pyo.Constraint(m.t, rule = TotalDischargeFx2)
    m.TotalDischarge3 = pyo.Constraint(m.t, rule = TotalDischargeFx3)
    m.TotalDischarge4 = pyo.Constraint(m.t, rule = TotalDischargeFx4)

    m.VolCons = pyo.Constraint(rule = VolumeConservation)

    m.InitLevel = pyo.Constraint(rule = LevelInitFx)
    m.InitOF = pyo.Constraint(rule = OverflowInitFx)
    m.InitDischarge = pyo.Constraint(m.Outlets, rule = DischargeInitFx)
    m.BoundLevel = pyo.Constraint(m.t, rule = LevelBounds)
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

def CompileModelParams(P, n_timesteps = 288, InitLevels = None, a = None, b = None):
    # Default values
    if a is None:
        alpha = 1.0
    else:
        alpha = a
    if b is None:
        beta = 1.0
    else:
        beta = b
    if InitLevels is None:
        InitialLevels = 0.0
    else:
        InitialLevels = InitLevels

    dct = {None: {
        't': {None: list(range(n_timesteps))}, # This can be changed
        'T': {None: list(range(1, n_timesteps))},
        'Outlets' : {None: ['SH', 'SD', 'SD2']},
        'CondOutlets' : {None: ['SH', 'SD2']},

        'alpha': {None: alpha},
        'beta' : {None: beta},

        # -----
        'OrificeArea': P['OrificeAreas'],
        'OrificeHeight': P['OrificeHeights'],
        'TankHeight': {None: P['TankHeights']},
        'TankArea': {None: P['TankAreas']},
        'InitialLevel': {None: InitialLevels},
      }}

    return dct

def SolveInstance(instance, solver = 'cplex', PrintSolverOutput = True, \
                    Gap = None, TimeLimit = 300):
    opt = pyo.SolverFactory(solver)

    # ----- Optimisation options
    if Gap is None:
        Gap = 1e-4
    opt.options['mipgap'] = Gap
    opt.options['timelimit'] = TimeLimit

    opt_results = opt.solve(instance, tee = PrintSolverOutput)

    return {'Instance': instance, 'SolverOutput': opt_results}


def CreateFakeDimensions():
    P = {'TankAreas': 20,
         'TankHeights': 2.0,
         'OrificeHeights': {'SH': 0.4, 'SD2': 1.0, 'SD': 0.0},
         'OrificeAreas': {'SH': 0.07068583470577035, 'SD': 0.007853981633974483, 'SD2': 0.5026548245743669}
         }

    return P

def CreateRandomDimensions():
    P = {'TankAreas': np.random.choice([5,10,15,20,25]),
         'TankHeights': np.random.choice([0.5, 1.0, 1.5, 2.0, 3.0]),
         'OrificeHeights': {'SH': 0.4, 'SD2': 1.0, 'SD': 0.0},
         'OrificeAreas': {'SH': 0.07068583470577035, 'SD': 0.007853981633974483, 'SD2': 0.5026548245743669}
         }

    return P


def Summarise(inst):
    Opt_Level = pd.DataFrame.from_dict(inst.Level.extract_values(), orient = 'index', columns = ['Level'])

    Opt_Discharge = pd.DataFrame.from_dict(inst.Discharge.extract_values(), orient = 'index', columns = ['Discharge'])
    Opt_Discharge.index = pd.MultiIndex.from_tuples(Opt_Discharge.index, names = ['Outlet', 'Timestep'])
    Opt_Discharge = Opt_Discharge.reset_index().pivot(index = 'Timestep', columns = 'Outlet', values = 'Discharge')

    Rainin = pd.DataFrame.from_dict(inst.RainIn.extract_values(), orient = 'index', columns = ['RainIn'])
    Opt_OF = pd.DataFrame.from_dict(inst.Overflow.extract_values(), orient = 'index', columns = ['Overflow'])
    Total_Discharge = pd.DataFrame.from_dict(inst.TotalDischarge.extract_values(), orient = 'index', columns = ['Total Discharge'])

    results = pd.concat([Rainin, Opt_Level, Opt_Discharge, Opt_OF,  Total_Discharge], axis = 1)

    return results

def PlotDynamics(Dataframe, instance):
    plt.figure()
    colors = sns.color_palette()

    plt.subplot(311)
    plt.plot(Dataframe['RainIn'], label = 'Rain')
    plt.plot(Dataframe['Overflow'], label = 'Overflow S')
    plt.legend()

    # ----- Plot Tank Dynamics
    plt.subplot(312)
    cnt = 0
    plt.plot(Dataframe.Level, label = 'Separation Tank Levels', color = colors[cnt])
    plt.axhline(instance.TankHeight(), xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    plt.axhline(instance.OrificeHeight.extract_values()['SH'], xmin = 0, xmax = 1, color = colors[0], linestyle = '--', alpha = 0.5)
    plt.axhline(instance.OrificeHeight.extract_values()['SD2'], xmin = 0, xmax = 1, color = colors[2], linestyle = '--', alpha = 0.5)
    cnt +=1
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
    plt.legend()
    plt.ylabel('Discharge Volumes \n per timestep [m$^3$/s]')
    plt.xlabel('Simulation Timestep ID')

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


# ========== WRAPPER FUNCTIONS
