import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


output_specification_agent_prompt = """
# ✅ **NGSpice `.measure` Syntax Explained**

The `.measure` (or `.meas`) statement tells NGSpice to compute a **scalar value** from the simulation results — duration, voltage, current, power, peak values, crossing times, integrals, etc.

General form:

```
.measure <analysis> <name> <type> <conditions...>
```

Where:

* `<analysis>` = `tran`, `ac`, or `dc`
* `<name>` = user-defined measurement variable, e.g. `tpeak`, `imax`
* `<type>` = `PARAM`, `MAX`, `MIN`, `AVG`, `INTEG`, or omitted for WHEN/CROSS

---

# 🔶 **1. WHEN — value at a specific event**

Use WHEN when you want the **time at which** a signal first equals a value.

Syntax:

```
.measure tran t1 WHEN V(node)=value
```

Example:

```
.measure tran tzero WHEN I(L1)=0 CROSS=2
```

Meaning:

* Find the **second** time that inductor current crosses zero
* Return that timestamp
* This is used for conduction-time or commutation problems

You can add qualifiers:

* `CROSS=n` = nth crossing
* `RISE=n` = nth rising edge
* `FALL=n` = nth falling edge

If you omit them, NGSpice uses the **first** occurrence.

---

# 🔶 **2. MAX / MIN — peak values**

```
.measure tran imax MAX I(L1)
.measure tran vmin MIN V(out)
```

Meaning:

* Maximum of I(L1) over the transient interval
* Minimum of V(out)

You may also restrict the interval:

```
.measure tran imax MAX I(L1) FROM=0 TO=50u
```

---

# 🔶 **3. AVG — average value**

```
.measure tran vp_avg AVG V(phase) FROM=0 TO=10ms
```

Computes time average over an interval.

---

# 🔶 **4. INTEG — integrate over time**

```
.measure tran energy INTEG P(R1) FROM=0 TO=1ms
```

Useful for energy, charge, or magnetic flux calculations.

---

# 🔶 **5. PARAM — compute expression from other measures**

```
.measure tran power PARAM='imax * vpeak'
```

Allows algebraic expressions based on measured values.

---

# 🔶 **6. Using FROM and TO**

All `.measure` statements can restrict time or frequency window:

```
.measure tran t90 WHEN V(out)=0.9 FROM=0us TO=20us
```

---

# 🔶 **7. Combining Types — common examples**

### ◎ Find frequency of an LC oscillator:

```
.measure tran t1 WHEN V(out)=0 CROSS=1
.measure tran t2 WHEN V(out)=0 CROSS=3
.measure tran freq PARAM='1/(t2-t1)'
```

### ◎ Peak-to-peak voltage:

```
.measure tran vmax MAX V(out)
.measure tran vmin MIN V(out)
.measure tran vpp PARAM='vmax - vmin'
```

### Delay between signals:

```
.measure tran tA WHEN V(a)=2.5 RISE=1
.measure tran tB WHEN V(b)=2.5 RISE=1
.measure tran delay PARAM='tB - tA'
```
# Power factor / cos(phi) measurement (AC steady state)
If the question asks for power factor, cos(phi), phase angle, or "leading/lagging":
- Do NOT measure only node voltages.
- Measure source current and compute cos(phi) using complex power.

Use these statements (for AC analysis):
.measure ac Vrms  RMS V(<source_pos>,<source_neg>)
.measure ac Irms  RMS I(<source_name>)
.measure ac Pavg  AVG  ( V(<source_pos>,<source_neg>) * I(<source_name>) )
.measure ac pf    PARAM='Pavg/(Vrms*Irms)'

Choose the single frequency point used by the analysis (.ac lin 1 {f} {f} or equivalent).

# DC operating point quantities
If the question asks for VCE (or UCE, V_CE) or quiescent point (Q-point):
- Measure it explicitly as a difference of node voltages:
.measure op vce PARAM='V(<collector_node>)-V(<emitter_node>)'

If the question asks for IC (or I_CQ):
- Prefer measuring current through the collector resistor or the supply branch current (sign may need interpretation):
.measure op ic  PARAM='-I(<VCC_source_name>)'

# If the question mentions "voltage across load/appliance",
measure V(load_pos) − V(load_neg), not a nearby voltmeter.


# 🔶 **Example**
Question:
Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: In the figure below, Thyristor T is initially off and is triggered with a single pulse of width \( 10 \ \mu s \). It is given that \( L = \left(\frac{100}{\pi}\right) \ \mu H \) and \( C = \left(\frac{100}{\pi}\right) \ \mu F \). Assuming latching and holding currents of the thyristor are both zero and the initial charge on C is zero, T conducts for ( ) \ \mu s.

Necessary measurement statement to find conduction time:
```
.measure tran tcond WHEN I(L1)=0 CROSS=2
```

Interpretation:

* Find the **second** time inductor current becomes zero
* The first crossing is at t=0 (start)
* The second crossing gives the conduction duration

---

# ✔ Summary Table

| Purpose               | Example                                   |
| --------------------- | ----------------------------------------- |
| Time of event         | `.measure tran t1 WHEN V(out)=1`          |
| nth crossing          | `.measure tran t2 WHEN I(L1)=0 CROSS=2`   |
| Max/Min               | `.measure tran imax MAX I(L1)`            |
| Average               | `.measure tran vavg AVG V(out)`           |
| Integrate             | `.measure tran e INTEG P(R1)`             |
| Parameter expressions | `.measure tran freq PARAM='1/(t_period)'` |

---

Given the ngspice program and the question, decide if a measurement statement is needed to answer the question.
# Question:
Hint: Please answer the question requiring an integer answer.
Question: In the figure below, Thyristor T is initially off and is triggered with a single pulse of width 10 μs. It is given that
L = (100/π) μH and C = (100/π) μF. Assuming latching and holding currents of the thyristor are both zero and the initial charge on C is zero, T conducts for ( ) μs.

# Domain knowledge:
Here is an explanation of the key vocabulary and terms related to electrical engineering in the given question:

1. **Thyristor (T)**: A semiconductor device that acts as a switch, conducting current only after being triggered by a gate pulse. It remains conducting even if the gate signal is removed, until the current through it falls below a certain level.

2. **Initially off**: The thyristor is initially in a non-conducting state before the gate trigger is applied.

3. **Triggered with a single pulse of width \(10 \ \mu s\)**: The thyristor is turned on by a gate pulse that lasts for 10 microseconds (\(\mu s\)).

4. **Pulse width**: Duration of the triggering pulse.

5. **\( L \)**: Inductance, a property of an electrical circuit or component that opposes changes in current. It is given as \( \left(\frac{100}{\pi}\right) \ \mu H \) (microhenries).

6. **Inductance (\(\mu H\))**: Unit of inductance; \(1 \ \mu H = 10^{-6} \ H\).

7. **\( C \)**: Capacitance, a property of a component that stores electrical energy in an electric field, given as \( \left(\frac{100}{\pi}\right) \ \mu F \) (microfarads).

8. **Capacitance (\(\mu F\))**: Unit of capacitance; \(1 \ \mu F = 10^{-6} \ F\).

9. **Latching current**: The minimum current required for the thyristor to latch into the "on" state and continue conducting after being triggered.

10. **Holding current**: The minimum current that must flow through the thyristor to keep it in the "on" state. If the current falls below this value, the thyristor turns off.

11. **Initial charge on C is zero**: The capacitor is initially uncharged before the operation begins.

12. **Conducts for ( ) \ \mu s**: Asking for the duration (in microseconds) that the thyristor remains in the conducting state after being triggered.

# Netlist:
* Thyristor LC conduction time, modeled as a step at t=0
.param Lval = 100e-6/3.1416
.param Cval = 100e-6/3.1416
V1 Nsrc 0 PULSE(0 15 0 1n 1n 1 1m)
R1 Nsrc N1 1m          ; tiny series resistance
L1 N1   N2 {Lval}
C1 N2   0  {Cval}
.tran 1u 200u
.print tran I(L1)
.end

# Output:
Reasoning:
- The question asks for the conduction time of the thyristor T, which is the duration from when it turns on until the current through it returns to zero.
- This can be determined by measuring the time at which the current through the inductor L1 (which is in series with the thyristor) crosses zero for the second time (the first crossing is at t=0 when it turns on).
- Therefore, we need a `.measure` statement that captures this event.

edit:
section: analysis
- name: .measure, type: analysis_type, old: none, new: .measure tran Tcond WHEN I(L1)=0 CROSS=2

if no measurement is needed:
# Output:
Reasoning:
- After analyzing the question and the provided ngspice program, it is determined that no additional measurement statement is necessary to answer the question.

edit:
"""
