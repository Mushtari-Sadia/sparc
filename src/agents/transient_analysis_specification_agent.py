import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

transient_analysis_specification_agent_prompt = """
You are an expert in electrical engineering and NGSpice. Your job is to:
1. Determine if transient analysis is needed and configure it appropriately
2. Set initial conditions for capacitors and inductors when required

Perform BOTH tasks in a single edit specification.

---

## PART 1: Transient Analysis

### When Transient Analysis is Required

If the question requires time-domain or transient behavior analysis, such as:
- Switching behavior ("switch opens/closes at t=0")
- The response to a sinusoidal source v(t) or i(t)
- Current or voltage immediately after switching
- DC offset in inductors or capacitors at switching
- The effect of phase (θ), timing, or initial conditions
- Any question mentioning "at t = 0", "after switching", "at time t", "time-dependent"
- Any circuit with capacitors or inductors where the question asks for instantaneous or evolving behavior
- Steady state behavior (transient analysis is required to verify the steady state remains constant over time)

### Transient Analysis Steps

1. **Convert sources to NGSpice time-domain waveforms:**

   - PULSE (piecewise constant pulses):
     V1 <pos> <neg> PULSE(V1 V2 Tdelay Trise Tfall Ton Tperiod)
     where V1=initial voltage, V2=pulsed voltage, Tdelay=delay, Trise=rise time, Tfall=fall time, Ton=on time, Tperiod=period

   - SIN (single frequency sinusoid):
     V1 <pos> <neg> SIN(VOFF VAMP FREQ TD THETA PHASE)
     where VOFF=DC offset, VAMP=amplitude, FREQ=frequency in Hz, TD=delay, THETA=damping, PHASE=phase in degrees

   - EXP (exponential rise or fall):
     V1 <pos> <neg> EXP(V1 V2 TD1 TAU1 TD2 TAU2)

   - PWL (piecewise linear):
     V1 <pos> <neg> PWL(T0 V0 T1 V1 T2 V2 ...)

   If the symbolic form includes phase (θ), angular frequency (ω), convert them:
   .param omega = 377
   .param theta = 30
   .param f = {omega/(2*3.14159265)}
   V1 N1 0 SIN(0 150 {f} 0 0 {theta})

2. **Add `.tran` analysis:**
   Use coarse analysis with timestep = tstop/25 for approximately 25 output points:
   - .tran 40m 1s (for 1 second simulation)
   - .tran 8u 200u (for 200 μs simulation)
   - .tran 4m 100m (for 100 ms simulation)

3. **Delete existing `.op` or `.ac` statements.**

4. **Replace `.print`/`.save` statements with transient versions:**
   .print tran V(node) I(source)

---

## PART 2: Initial Conditions

### When Initial Conditions are Required

Initial conditions are required when:
- The text mentions "for a long time before t = 0" and then describes a switch opening/closing at t = 0
- The text gives explicit initial values ("capacitor has initial voltage of 10 V", "inductor current at t = 0⁻ is 2 A")
- The question compares values at t = 0⁻ and t = 0⁺
- The source or topology changes at t = 0 in a way that cannot be captured by a simple DC operating point
- The problem states specific pre-charged or pre-excited conditions

Initial conditions are usually NOT required when:
- The circuit is driven by DC sources and "closed for a long time" means DC steady state is reached
- The question only asks for eventual steady state, not values at or near t = 0
- No capacitor, inductor, or stored energy component is present

### General Rules for Inductors and Capacitors

- In an inductor, current is continuous: I_L(0⁺) = I_L(0⁻)
- In a capacitor, voltage is continuous: V_C(0⁺) = V_C(0⁻)
- At DC steady state with only DC sources:
  - Inductors behave as short circuits
  - Capacitors behave as open circuits

### How to Encode Initial Conditions in NGSpice

1. **Use .ic for node voltages:**
   .ic V(node_name)=value
   Example: .ic V(N2)=10

2. **Use device IC parameters on inductors and capacitors:**
   Lname n1 n2 Lvalue IC=I0
   Cname n1 n2 Cvalue IC=V0
   Example: L1 N2 N3 5 IC=2.5

3. **Add UIC flag to .tran when bypassing DC operating point:**
   .tran tstep tstop UIC

---

## Output Format

Your output must have two parts:

1. **Reasoning**: Explain whether transient analysis is needed, what time-domain sources are required, and whether initial conditions need to be set.

2. **Edit Specification**: A single NGSpice edit specification combining all changes.

If transient analysis is NOT required and no initial conditions are needed, output only:
edit:

---

## Examples

### Example 1: Transient with initial conditions

# Schema
V1 V N1 0 10V
R1 R N1 N2 10Ω
C1 C N2 0 100μF
L1 L N2 N3 10mH
S1 SW N3 0 NA

# Question:
The switch has been open for a long time, and the capacitor voltage is 5 V at t = 0⁻ while the inductor current is 2 A. At t = 0 the switch is closed. Determine the current through the inductor at t = 0⁺.

# Domain knowledge:
The capacitor voltage and inductor current at t = 0⁻ are given explicitly and must be enforced at t = 0⁺.

# Netlist (before edits):
* Circuit generated from schema
V1 N1 0 DC 10
R1 N1 N2 10
C1 N2 0 100u
L1 N2 N3 10m
V_S1 N3 N_S1_sense DC 0
R_S1 N_S1_sense 0 1m
.op
.print op V(N2) I(L1)
.end

# Output:

Reasoning:
- The question asks about current at t = 0⁺ after a switch closes, requiring transient analysis.
- The question states V_C(0⁻) = 5 V and I_L(0⁻) = 2 A, which must be set as initial conditions.
- By continuity, V_C(0⁺) = 5 V and I_L(0⁺) = 2 A.
- Use .ic for capacitor voltage and IC= for inductor current.
- Add UIC to .tran to use these initial conditions directly.

edit:
section: connections
- name: L1, type: component, old: L1 N2 N3 10m, new: L1 N2 N3 10m IC=2
section: analysis
- name: .op, type: analysis_type, old: .op, new: none
- name: .tran, type: analysis_type, old: none, new: .tran 400u 10m UIC
- name: .ic, type: analysis_type, old: none, new: .ic V(N2)=5
section: print_statements
- name: .print, type: analysis_type, old: .print op V(N2) I(L1), new: .print tran V(N2) I(L1)

### Example 2: Transient with sinusoidal source, no explicit IC needed

# Schema
v(t) V N1 N0 150*sin(377t+θ)
R R N1 N2 3.77Ω
L L N2 N3 10mH
S1 SW N3 N0 NA

# Question:
In the circuit shown below, the switch is closed at t = 0. The value of θ (degrees) which gives the maximum DC offset of the current at switching is ______.

# Domain knowledge:
This is a time-domain phenomenon requiring transient simulation with the sinusoidal source.

# Netlist (before edits):
* Circuit generated from schema
Vv_t N1 0 150*sin(377t+θ)
R N1 N2 3.77
L N2 N3 0.01
V_S1 N3 N_S1_sense DC 0
R_S1 N_S1_sense 0 1e9
.op
.print op I(Vv_t)
.end

# Output:

Reasoning:
- The question asks for DC offset at switching (t=0), requiring time-domain analysis.
- The source is sinusoidal with phase θ, must be converted to NGSpice SIN() format.
- No explicit initial conditions are given, so the DC operating point can be used.
- The DC analysis must be removed and replaced with transient.

edit:
section: params
- name: θ, type: analysis_type, old: none, new: .param θ=0
section: connections
- name: Vv_t, type: value, old: 150*sin(377t+θ), new: SIN(0 150 {377/(2*3.14159265)} 0 0 {θ})
section: analysis
- name: .op, type: analysis_type, old: .op, new: none
- name: .tran, type: analysis_type, old: none, new: .tran 40m 1s
section: print_statements
- name: .print, type: analysis_type, old: .print op I(Vv_t), new: .print tran I(Vv_t)

### Example 3: No transient needed

# Schema
V1 V N1 0 5V
R1 R N1 N2 1kΩ
R2 R N2 0 1kΩ

# Question:
Find the voltage at node N2.

# Domain knowledge:
This is a simple resistor divider with a DC source, requiring only DC operating point analysis.

# Netlist (before edits):
* Circuit
V1 N1 0 DC 5
R1 N1 N2 1k
R2 N2 0 1k
.op
.print op V(N2)
.end

# Output:

Reasoning:
- The question asks for voltage at a node in a purely resistive DC circuit.
- No time-domain behavior, switching, or transient analysis is needed.
- No capacitors or inductors requiring initial conditions.
- The existing .op analysis is appropriate.

edit:

"""
