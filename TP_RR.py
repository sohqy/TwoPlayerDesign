"""
RR Model.

"""

# -----
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns

# ----- EQUATION DEFINITIONS
def ObjFx(m):
    """ """
    return sum(m.Overflow['S', t] for t in m.T)

# ----- Flow directions
def Inflow(m, j, t):
    if j == 'S':
        return m.RainIn[t] + sum(m.Overflow[j, t] for j in m.OFTanks)
    elif j == 'D':
        return m.Discharge['SD', t] + m.Discharge['SD2', t]
    elif j == 'H':
        return m.Discharge['SH', t]
    elif j == 'T':
        return m.Discharge['HT', t]

def Outflow(m, j, t):
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

    if j == 'T':
        return m.Level[j, t+1] == m.Level[j, t] + (\
                m.alpha[j] * Inflow(m, j, t) + (1 - m.alpha[j]) * Inflow(m, j, t+1) \
                - m.beta[j] * Outflow(m, j, t) - (1 - m.beta[j]) * Outflow(m, j, t+1) \
                ) / m.TankArea[j]
    else:
        return m.Overflow[j, t+1] / m.TankArea[j] + m.Level[j, t+1] == m.Level[j, t] + ( \
                m.alpha[j] * Inflow(m, j, t) + (1 - m.alpha[j]) * Inflow(m, j, t+1) \
                - m.beta[j] * Outflow(m, j, t) - (1 - m.beta[j]) * Outflow(m, j, t+1) \
                ) / m.TankArea[j]

# -----
def FreshwaterFx(m, t):
    return m.Demand[t] == m.Freshwater[t] + m.Discharge['TD', t]

def FreshwaterCostFx(m, t):
    return m.CostF[t] == m.Freshwater[t] * m.FreshwaterCost[t]

# -----
def OverflowFx1(m, j, t):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[j, t+1] >= m.TankArea[j] * m.Level[j, t] \
                + Inflow(m, j, t) - Outflow(m, j, t) - m.TankArea[j] * m.TankHeight[j]

def OverflowFx2(m, j, t):
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[j, t+1] <= m.TankArea[j] * m.Level[j, t]  \
                + Inflow(m, j, t) - Outflow(m, j, t) - m.TankArea[j] * m.TankHeight[j] \
                - m.BigM * m.OverflowBinary[j, t] + m.BigM

def OverflowEpsUB(m, j, t):
    if j == 'S':
        UpperBound = m.Epsilon
    else:
        UpperBound = m.BigM

    return m.Overflow[j, t] <= UpperBound * m.OverflowBinary[j, t]

# ----- Conditional flows
def CondFlowLower(m, c, t):
    if c in ['SD2', 'SH']:
        source = 'S'
    elif c == 'DO2':
        source = 'D'

    return m.Discharge[c, t] >= 2.66 * m.OrificeArea[c] * (m.Level[source, t] - m.OrificeHeight[c]) * m.DeltaT

def CondFlowUpper1(m, c, t):
    if c in ['SD2', 'SH']:
        source = 'S'
    elif c == 'DO2':
        source = 'D'

    return m.Discharge[c, t] <= 2.66 * m.OrificeArea[c] * m.DeltaT * \
            m.CondBinary[c, t] * (m.TankHeight[source] - m.OrificeHeight[c])

def CondFlowUpper2(m, c, t):
    if c in ['SD2', 'SH']:
        source = 'S'
    elif c == 'DO2':
        source = 'D'

    return m.Discharge[c, t] <= 2.66 * m.OrificeArea[c] * m.DeltaT * (m.Level[source, t] - m.OrificeHeight[c]) \
        - 2.66 * m.OrificeArea[c] * m.DeltaT * m.OrificeHeight[c] * m.CondBinary[c, t] \
        + 2.66 * m.OrificeArea[c] * m.DeltaT * m.OrificeHeight[c]

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

# ----- Vol Conservation


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


# -----
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
    WindowMin = int(t - m.WindowSize)
    if WindowMin <= m.WindowSize: # Only start calculating when we can have full windows.
        return pyo.Constraint.Skip
    else:
        Window = list(range(WindowMin, t+1))
        return sum(m.RainIn[w] for w in Window) <= m.RainWindowMax

def SetRainFinal(m, t):
    if t == 0:
        return m.RainIn[t] == 0.0
    else:
        return pyo.Constraint.Skip

def RainInBounds(m, t):
    return m.RainIn[t] <= m.RainTSMax


#%%

def CreateAbstractModel(DeltaT = 300, CatchmentSize = 15.7, Eps = 5000):
    m = pyo.AbstractModel()

    # ----- SETS
    m.t = pyo.Set(ordered = True, doc = 'Discretized Simulation Timesteps')
    m.T = pyo.Set(ordered = True, doc = 'Obj Fx Included Timesteps [1, ..., 288]')
    m.Tanks = pyo.Set(doc = 'Tank Names')
    m.OFTanks = pyo.Set(doc = 'Tanks with OF.')
    m.Outlets = pyo.Set(doc = 'Outlets within the system.')
    m.CondOutlets = pyo.Set(doc = 'Conditional flow outlets.')

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
    m.BigM = 5000

    m.HTOnPt = 1.0
    m.HTOffPt = 0.3

    m.UpperR = 11.4/DeltaT * CatchmentSize
    m.LowerR = -8.8/DeltaT * CatchmentSize
    m.WindowSize = 12.0 * 2           # 2 hours.
    m.RainHorizonMax = 181.2 * CatchmentSize
    m.RainWindowMax = 100 * CatchmentSize
    m.RainTSMax = 20 * CatchmentSize # historical observation = 14.6

    # ----- VARIABLES
    # Binary Variables
    m.CondBinary = pyo.Var(m.CondOutlets, m.t, within = pyo.Binary)
    m.OverflowBinary = pyo.Var(m.OFTanks, m.t, within = pyo.Binary)
    m.RelayBinary = pyo.Var(m.r, m.t, within = pyo.Binary)

    # Continuous Variables
    m.Level = pyo.Var(m.Tanks, m.t, within = pyo.NonNegativeReals)
    m.Overflow = pyo.Var(m.OFTanks, m.t, within = pyo.NonNegativeReals)
    m.Discharge = pyo.Var(m.Outlets, m.t, within = pyo.NonNegativeReals)

    m.Freshwater = pyo.Var(m.t, within = pyo.NonNegativeReals)
    m.CostF = pyo.Var(m.t, within = pyo.NonNegativeReals)
    m.DischargeLimit = pyo.Var(m.Outlets, within = pyo.NonNegativeReals)

    # Decision Variable
    m.RainIn = pyo.Var(m.t, within = pyo.NonNegativeReals)

    # ----- EQUATIONS
    m.Obj = pyo.Objective(rule = ObjFx, sense = pyo.maximize)

    # Tank Dynamics
    m.MassBalances = pyo.Constraint(m.Tanks, m.t, rule = MassBalancesFX)
    m.FreshwaterUse = pyo.Constraint(m.t, rule = FreshwaterFx)
    m.Cost = pyo.Constraint(m.t, rule = FreshwaterCostFx)

    m.Overflows1 = pyo.Constraint(m.OFTanks, m.t, rule = OverflowFx1)
    m.Overflows2 = pyo.Constraint(m.OFTanks, m.t, rule = OverflowFx2)
    m.Overflows3 = pyo.Constraint(m.OFTanks, m.t, rule = OverflowEpsUB)

    m.CondFlow1 = pyo.Constraint(m.CondOutlets, m.t, rule = CondFlowLower)
    m.CondFlow2 = pyo.Constraint(m.CondOutlets, m.t, rule = CondFlowUpper1)
    m.CondFlow3 = pyo.Constraint(m.CondOutlets, m.t, rule = CondFlowUpper2)

    m.DOFlow = pyo.Constraint(m.t, rule = DOFlowFx)
    m.SDFlow = pyo.Constraint(m.t, rule = SDFlowFx)

    m.RB = pyo.Constraint(m.t, rule = RelayBinariesFX)
    m.HT1 = pyo.Constraint(m.t, rule = HTRelay1)
    m.HT2 = pyo.Constraint(m.t, rule = HTRelay2)
    m.HT3 = pyo.Constraint(m.t, rule = HTRelay3)
    m.HT4 = pyo.Constraint(m.t, rule = HTRelay4)
    m.HT5 = pyo.Constraint(m.t, rule = HTRelay5)
    m.HT6 = pyo.Constraint(m.t, rule = HTRelay6)

    m.InitLevel = pyo.Constraint(m.Tanks, rule = LevelInitFx)
    m.InitOF = pyo.Constraint(m.OFTanks, rule = OverflowInitFx)
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
    m.FinalRain = pyo.Constraint(m.t, rule = SetRainFinal)
    m.BoundRain = pyo.Constraint(m.t, rule = RainInBounds)

    return m

def CompileModelParams(Parameters, DemandArray, Cost = None, InitialLevels = None, alpha = None, beta = None):
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

    # -----
    if alpha is None:
        alpha = {'S': 1.0, 'D': 1.0, 'H': 1.0, 'T': 1.0}

    if beta is None:
        beta = {'S': 1.0, 'D': 1.0, 'H': 1.0, 'T': 1.0}

    # ----- Compile data
    dct = {None: {
        't': {None: list(range(289))}, # This can be changed
        'T': {None: list(range(1, 289))},
        'Tanks': {None: ['S', 'H', 'D', 'T']},
        'OFTanks': {None: ['S', 'H', 'D']},
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
    #ObjVal = pyo.value(instance.Obj)

    #return {'Instance': instance, 'ObjValue': ObjVal, 'SolverOutput': opt_results}
    return {'Instance': instance, 'SolverOutput': opt_results}


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

    results = pd.concat([Rainin, Opt_Level, Opt_Discharge, Opt_OF], axis = 1)

    return results

def PlotDynamics(Dataframe, instance):
    plt.figure()
    colors = sns.color_palette()

    plt.subplot(311)
    plt.plot(Dataframe['RainIn'], label = 'RuinousRain')
    plt.plot(Dataframe['OF S'], label = 'Overflow S')
    plt.plot(Dataframe['OF D'], label = 'Overflow D')
    plt.plot(Dataframe['OF H'], label = 'Overflow H')
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


# def SepBalanceFx(m, t):
#     if t == m.t[-1]:
#         return pyo.Constraint.Skip
#
#     else:
#         def In_S(t):
#             return m.RainIn[t] + sum(m.Overflow[j, t] for j in m.Tanks)
#         def Out_S(t):
#             return m.Dicharge['SD', t] + m.Discharge['SD2', t] + m.Discharge ['SH'. t]
#
#         return m.Overflow['S', t+1]/m.TankArea['S'] + m.Level['S', t+1] == m.Level['S', t] + ( \
#                 m.alpha['S'] * In_S(t) + (1 - m.alpha['S']) * In_S(t+1) \
#                 - m.beta['S'] * Out_S(t) - (1 - m.beta['S']) * Out_S(t+1) \
#                 ) / m.TankArea['S']
#
# def DetBalanceFX(m, t):
#     if t == m.t[-1]:
#         return pyo.Constraint.Skip
#     else:
#         def In_D(t):
#             return m.Discharge['SD', t] + m.Discharge['SD2', t]
#         def Out_D(t):
#             return m.Discharge['DO', t] + m.Discharge['DO2', t]
#
#         return m.Overflow['D', t+1]/m.TankArea['D'] + m.Level['D'. t+1] == m.Level['D', t] + ( \
#                 m.alpha['D'] * In_D(t) + (1 - m.alpha['D']) * In_D(t+1) \
#                 - m.beta['D'] * Out_D(t) - (1 - m.beta['D']) * Out_D(t+1) \
#                 ) / m.TankArea['D']
#
# def HarBalanceFx(m, t):
#     if t == m.t[-1]:
#         return pyo.Constraint.Skip
#     else:
#         def In_H(t):
#             return m.Discharge['SH', t]
#         def Out_H(t):
#             return m.Discharge['HT', t]
#         return m.Overflow['H', t+1]/m.TankArea['H'] + m.Level['H', t+1] == m.Level['H', t] \
#         + (m.alpha['H'] * m.Discharge['SH', t] \
#             + (1 - m.alpha['H']) * m.Discharge['SH', t+1] \
#             - m.beta['H'] * m.Discharge['HT', t] \
#             - (1 - m.beta['H']) * m.Discharge['HT', t+1]
#         ) / m.TankArea['H']
