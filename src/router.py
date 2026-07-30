import sys
import os
from typing import Optional, TypedDict
import json
from model import *
from creator import *
from state import *
from agents.planner_agent import *
from agents.planner_agent_mcq import *
from agents.circuit_specification_agent import *
from agents.analysis_specification_agent import *
from agents.transient_analysis_specification_agent import *
from agents.output_specification_agent import *
from agents.error_diagnosis_agent import *
from agents.answer_generation_agent import *
from make_update_agent import *
import time
class Router:
    def __init__(self, config, model):
        self.model = model
        self.config = config
        self.netlist = config.get('netlist', '')
        if not self.netlist:
            creator = Creator(config['schema'])
            template = creator.make_template()
            self.state = NgSpiceState(self.model, creator)
        else:
            self.state = NgSpiceState(self.model)
            self.state.netlist_text = self.netlist
        self.state.index = config.get('index', -1)
        self.state.question = config.get('question', '')
        self.state.schema = config.get('schema', '')
        self.state.image = config.get('image', '')
        self.state.domain_knowledge = config.get('domain_knowledge', '')
        self.max_retries = config.get('max_retries', 3)
        self.self_consistency = config.get('self_consistency', 1)

    def route(self,mcq_enabled=False):
        
        if mcq_enabled:
            simulation_states = planner_agent_mcq(self.state)
        else:
            simulation_states = planner_agent(self.state)


        for sim_state in simulation_states:
            # Static sequence for first run
            sim_state = self._run_static_sequence(sim_state)
            
            # Dynamic error recovery loop
            retry_count = 0
            while sim_state.last_error and retry_count < self.max_retries:
                print(f"Error detected. Attempting recovery {retry_count + 1}/{self.max_retries}...")
                
                # Analyze the error
                sim_state = error_diagnosis_agent(sim_state)

                # Get dynamic agent sequence from model
                recovery_sequence = self._get_recovery_sequence(sim_state)
                
                # Execute the recovery sequence
                sim_state = self._execute_recovery_sequence(sim_state, recovery_sequence)
                
                
                retry_count += 1
            
            if sim_state.last_error:
                # print(f"Failed to recover after {self.max_retries} attempts.")
                return -1
        
        self.states = simulation_states
        return 0

    def _run_static_sequence(self, sim_state):
        """Execute the standard static sequence of agents."""
        # Combined values update and model addition
        # print("Agent: VALUES & MODELS (update values + add missing models)")
        sim_state = update_agent(sim_state, circuit_specification_agent_prompt, self_consistency_number=self.self_consistency)

        # # Old separate approach (commented out)
        # print("Agent: UPDATE VALUES")
        # sim_state = update_agent(sim_state, update_values_agent_prompt, self_consistency_number=1)
        # print("Agent: MODEL AGENT")
        # sim_state = update_agent(sim_state, model_agent_prompt)

        # Combined analysis classification and configuration
        # print("Agent: ANALYSIS (DC/AC classification + configuration)")
        sim_state = update_agent(sim_state, analysis_specification_agent_prompt, self_consistency_number=self.self_consistency)

        # # Old separate approach (commented out)
        # print("Agent: ANALYSIS CLASSIFIER")
        # sim_state = analysis_classifier_agent(sim_state)
        # # check in last 10 characters
        # if "dc" in sim_state.analysis_kind.lower()[-5:]:
        #     print("Agent: DC")
        #     sim_state = update_agent(sim_state, DC_prompt)
        # elif "ac" in sim_state.analysis_kind.lower()[-5:]:
        #     print("Agent: AC")
        #     sim_state = update_agent(sim_state, AC_prompt)
        
        
        # Combined transient analysis and initial conditions
        # print("Agent: TRANSIENT & IC (transient analysis + initial conditions)")
        # sim_state = update_agent(sim_state, transient_analysis_specification_agent_prompt)
        
        # # Old separate approach (commented out)
        # print("Agent: TRAN")
        # sim_state = update_agent(sim_state, transient_analysis_prompt)
        # print("Agent: INITIAL CONDITIONS")
        # sim_state = update_agent(sim_state, initial_conditions_agent_prompt)

        # Decide on sweep
        # print("Agent: SWEEP")
        # sim_state = update_agent(sim_state, sweep_prompt)
        # print("Agent: MEASUREMENT")
        sim_state = update_agent(sim_state, output_specification_agent_prompt, self_consistency_number=self.self_consistency)

        # Run NGSpice
        sim_state = run_ngspice_node(sim_state)
        
        return sim_state

    def _get_recovery_sequence(self, sim_state):
        """Use the model to determine which agents to call for error recovery."""
        system_prompt = """
You are an expert at diagnosing and fixing NGSpice simulation errors. Based on the error analysis, 
you must determine which agents should be called to fix the problem.

Available agents:
- circuit_specification_agent: Update component values and add/correct .model statements for devices (BJT, MOSFET, diode models)
- analysis_specification_agent: Classify and configure analysis type (DC or AC) with appropriate sources and statements
- transient_analysis_specification_agent: Configure transient analysis (.tran) with initial conditions for capacitors/inductors
- output_specification_agent: Modify .print/.save statements to capture the correct outputs

Return a JSON array of agent names in the order they should be executed.
Example: ["circuit_specification_agent", "analysis_specification_agent", "output_specification_agent"]

Only include agents that are necessary to fix the identified error.
"""

        user_prompt = f"""
Question: {sim_state.question}

Schema: {sim_state.schema}

Current Netlist:
{sim_state.netlist_text}

NGSpice Error Output:
{sim_state.ngspice_stderr}

Error Analysis:
{sim_state.error_analysis}

Based on the error analysis above, determine which agents should be called to fix the error.
Return ONLY a JSON array of agent names, nothing else.
"""

        response = self.model.predict(system_prompt=system_prompt, user_prompt=user_prompt)
        # print("=== Agent Output ===")
        # print(response)
        
        # Parse the JSON response
        try:
            # Try to extract JSON array from response
            response = response.strip()
            if not response.startswith('['):
                # Try to find JSON array in the response
                start = response.find('[')
                end = response.rfind(']') + 1
                if start != -1 and end > start:
                    response = response[start:end]
            
            recovery_sequence = json.loads(response)
            return recovery_sequence
        except json.JSONDecodeError as e:
            # print(f"Failed to parse recovery sequence: {e}")
            # print(f"Response was: {response}")
            # Fallback to a safe default
            return ["circuit_specification_agent", "output_specification_agent"]

    def _execute_recovery_sequence(self, sim_state, recovery_sequence):
        """Execute the dynamically determined sequence of agents."""
        agent_map = {
            "circuit_specification_agent": circuit_specification_agent_prompt,
            "analysis_specification_agent": analysis_specification_agent_prompt,
            "transient_analysis_specification_agent": transient_analysis_specification_agent_prompt,
            "output_specification_agent": output_specification_agent_prompt,
        }
        
        for agent_name in recovery_sequence:
            if agent_name in agent_map:
                # print(f"  Executing recovery agent: {agent_name}")
                sim_state = update_agent(sim_state, agent_map[agent_name])
                # rerun and check if error resolved
                sim_state = run_ngspice_node(sim_state)
                if not sim_state.last_error:
                    # run error analysis agent
                    # print("Error resolved.")
                    return sim_state
                else:
                    sim_state = error_diagnosis_agent(sim_state)
                

            else:
                # print(f"  Warning: Unknown agent '{agent_name}' in recovery sequence")
                pass
        
        return sim_state


            
    def solve(self):
        answer = answer_generation_agent(self.state.question, self.states, num_samples=self.self_consistency)
        # use answer for self consistency check
        return answer

        

