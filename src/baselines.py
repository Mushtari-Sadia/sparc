from utils import *
import subprocess
import tempfile
from typing import Optional, Tuple

# def baseline_1(model, schema, question):
#     system_prompt = "Given the schema and the question, provide the solution."
#     user_prompt = f"""Circuit schema:
# {schema}

# Question:
# {question}
# Provide the answer directly without any explanations. If the question is MCQ, provide only the letter of the correct choice.
# # Answer:
# """
#     output = model.predict(system_prompt, user_prompt)
#     
#     return output.strip()

# def baseline_2(model, schema, question, image):
#     system_prompt = "Given the diagram, the schema and the question, provide the solution."
#     user_prompt = f"""Circuit schema:
# {schema}

# Question:
# {question}
# Provide the answer directly without any explanations. If the question is MCQ, provide only the letter of the correct choice.
# # Answer:
# """
#     image, image_bytes = to_pil_and_bytes(image)
#     
#     output = model.predict_vl(system_prompt, user_prompt, images=[image])
#          
#     return output.strip()

# def baseline_3(model, question, image):
#     system_prompt = "Given the diagram and the question, provide the solution."
#     user_prompt = f"""Question:
# {question}
# Provide the answer directly without any explanations. If the question is MCQ, provide only the letter of the correct choice.
# # Answer:
# """
#     image, image_bytes = to_pil_and_bytes(image)
#     output = model.predict_vl(system_prompt, user_prompt, images=[image])
#     # print("Baseline Output:")
#     # print(output)
#     return output.strip()


def baseline_4_cot(model, question, schema):
    system_prompt = "Given the diagram schema and the question, do chain-of-thought reasoning of the problem, and provide the solution."
    user_prompt = f"""
Schema:
{schema}
    Question:
{question}
If the question is MCQ, provide only the letter of the correct choice. Provide the final answer in between the tags <final_answer>ANSWER</final_answer>:
"""
    output = model.predict(system_prompt, user_prompt, reasoning=False, max_tokens=1024)

    # parse the output to extract the final answer between the tags
    start_tag = "<final_answer>"
    end_tag = "</final_answer>"
    start_index = output.find(start_tag)
    end_index = output.find(end_tag)
    if start_index != -1 and end_index != -1:
        final_answer = output[start_index + len(start_tag):end_index].strip()
    else:
        final_answer = output.strip()
    
    return final_answer


def baseline_5_cot(model, question, schema, image):
    system_prompt = "Given the diagram, the schema and the question, do chain-of-thought reasoning of the problem, and provide the solution."
    user_prompt = f"""
Schema:
{schema}
    Question:
{question}
 If the question is MCQ, provide only the letter of the correct choice. Provide the final answer in between the tags <final_answer>ANSWER</final_answer>:
"""
    output = model.predict_vl(system_prompt, user_prompt, images=[image], reasoning=False, max_tokens=1024)

    # parse the output to extract the final answer between the tags
    start_tag = "<final_answer>"
    end_tag = "</final_answer>"
    start_index = output.find(start_tag)
    end_index = output.find(end_tag)
    if start_index != -1 and end_index != -1:
        final_answer = output[start_index + len(start_tag):end_index].strip()
    else:
        final_answer = output.strip()
    
    return final_answer

def baseline_6_cot(model, question, image):
    system_prompt = "Given the diagram and the question, do chain-of-thought reasoning of the problem, and provide the solution."
    user_prompt = f"""
    Question:
{question}
 If the question is MCQ, provide only the letter of the correct choice. Provide the final answer in between the tags <final_answer>ANSWER</final_answer>:
"""
    output = model.predict_vl(system_prompt, user_prompt, images=[image], reasoning=False, max_tokens=1024)

    # parse the output to extract the final answer between the tags
    start_tag = "<final_answer>"
    end_tag = "</final_answer>"
    start_index = output.find(start_tag)
    end_index = output.find(end_tag)
    if start_index != -1 and end_index != -1:
        final_answer = output[start_index + len(start_tag):end_index].strip()
    else:
        final_answer = output.strip()
    
    return final_answer


def baseline_7_sympy(model, question, schema, image=None):
    """
    Baseline that generates SymPy code to solve the problem and executes it.
    """
    system_prompt = """You are an expert in circuit analysis and symbolic mathematics. 
Generate Python code using SymPy to solve the given circuit problem.
Your code should:
1. Define all necessary symbols
2. Set up equations based on circuit laws (Ohm's law, KVL, KCL, etc.)
3. Solve the equations
4. Print the final answer

The code must be complete and executable. Wrap your code between ```python and ``` markers.
At the end, print the final answer with: print(f"FINAL_ANSWER: {{answer}}")"""

    user_prompt = f"""
Schema:
{schema}

Question:
{question}

Generate complete SymPy code to solve this problem. Include all necessary imports and make sure to print the final answer.
"""

    if image is not None:
        output = model.predict_vl(system_prompt, user_prompt, images=[image], reasoning=False, max_tokens=2048)
    else:
        output = model.predict(system_prompt, user_prompt, reasoning=False, max_tokens=2048)
    
    # Extract code between ```python and ```
    code = extract_code_from_output(output)
    
    if code is None:
        print("Warning: Could not extract code from model output")
        return output.strip()
    
    # Execute the code and get the answer
    answer = execute_sympy_code(code)
    
    return answer


def extract_code_from_output(output: str) -> Optional[str]:
    """Extract Python code from markdown code blocks."""
    import re
    
    # Try to find code between ```python and ```
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, output, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try without language specifier
    pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(pattern, output, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    return None


def execute_sympy_code(code: str, timeout: int = 30) -> str:
    """
    Execute SymPy code safely in a subprocess and return the final answer.
    
    Args:
        code: Python code string to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        The final answer extracted from the code output
    """
    # Create a temporary file with the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        # Execute the code in a subprocess with timeout
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout
        
        # Check for errors
        if result.returncode != 0:
            print(f"Code execution error:\n{result.stderr}")
            return "ERROR: Code execution failed"
        
        # Extract final answer
        final_answer = extract_final_answer(output)
        
        return final_answer
        
    except subprocess.TimeoutExpired:
        print(f"Code execution timed out after {timeout} seconds")
        return "ERROR: Timeout"
    except Exception as e:
        print(f"Error executing code: {e}")
        return f"ERROR: {str(e)}"
    finally:
        # Clean up temporary file
        import os
        try:
            os.unlink(temp_file)
        except:
            pass


def extract_final_answer(output: str) -> str:
    """Extract the final answer from code execution output."""
    import re
    
    # Look for FINAL_ANSWER: pattern
    pattern = r'FINAL_ANSWER:\s*(.+)'
    match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
    
    if match:
        return match.group(1).strip()
    
    # If no FINAL_ANSWER marker, return the last non-empty line
    lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
    if lines:
        return lines[-1]
    
    return output.strip()

