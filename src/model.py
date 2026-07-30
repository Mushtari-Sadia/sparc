from openai import AzureOpenAI, OpenAI
import warnings
warnings.filterwarnings('ignore')
import json
import math
import os
import base64
from io import BytesIO
import requests

import torch
try:
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel, LlavaForConditionalGeneration
except ImportError:
    AutoProcessor = AutoModelForCausalLM = AutoModel = LlavaForConditionalGeneration = None
from PIL import Image
import gc

from utils import *


# Optional: bitsandbytes for 4-bit fallback
try:
    import bitsandbytes as bnb  # noqa: F401
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False


def _img_to_data_url(img: Image.Image, fmt="PNG") -> str:
    """Convert a PIL image to a data URL string."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def _complex_from_any(x):
    """Convert a number or {real, imag} dict to complex."""
    if isinstance(x, dict):
        return complex(x["real"], x["imag"])
    return complex(x, 0.0)


def _complex_to_dict(z: complex):
    """Convert complex to {real, imag} dict."""
    return {"real": float(z.real), "imag": float(z.imag)}


def _ensure_pil_list(x):
    """Return a list of PIL Images."""
    import io
    import numpy as np
    
    if x is None:
        return []
    
    if isinstance(x, (list, tuple)):
        items = []
        for item in x:
            items.extend(_ensure_pil_list(item))
        return [im for im in items if im is not None]
    
    if isinstance(x, str):
        return [Image.open(x).convert("RGB")]
    
    if isinstance(x, (bytes, bytearray, memoryview)):
        return [Image.open(io.BytesIO(bytes(x))).convert("RGB")]
    
    if isinstance(x, Image.Image):
        return [x.convert("RGB")]
    
    if "numpy" in type(x).__module__:
        arr = np.array(x)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        return [Image.fromarray(arr.astype("uint8")).convert("RGB")]
    
    try:
        return _ensure_pil_list(bytes(x))
    except Exception:
        raise TypeError(f"Unsupported image type: {type(x)}")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ohms_law_voltage",
            "description": "Compute voltage V from resistance R and current I using V = I * R.",
            "parameters": {
                "type": "object",
                "properties": {
                    "R": {"type": "number", "description": "Resistance in ohms"},
                    "I": {"type": "number", "description": "Current in amperes"}
                },
                "required": ["R", "I"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ohms_law_current",
            "description": "Compute current I from voltage V and resistance R using I = V / R.",
            "parameters": {
                "type": "object",
                "properties": {
                    "V": {"type": "number", "description": "Voltage in volts"},
                    "R": {"type": "number", "description": "Resistance in ohms"}
                },
                "required": ["V", "R"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ohms_law_resistance",
            "description": "Compute resistance R from voltage V and current I using R = V / I.",
            "parameters": {
                "type": "object",
                "properties": {
                    "V": {"type": "number", "description": "Voltage in volts"},
                    "I": {"type": "number", "description": "Current in amperes"}
                },
                "required": ["V", "I"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "voltage_add",
            "description": "Add a list of voltages (real or complex).",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "real": {"type": "number"},
                                        "imag": {"type": "number"}
                                    },
                                    "required": ["real", "imag"]
                                }
                            ]
                        }
                    }
                },
                "required": ["values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "voltage_difference",
            "description": "Compute voltage drop v1 - v2, inputs may be real or complex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "v1": {
                        "description": "Voltage at first node",
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    },
                    "v2": {
                        "description": "Voltage at second node",
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    }
                },
                "required": ["v1", "v2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "current_add",
            "description": "Add a list of currents (real or complex), for KCL and other sums.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "real": {"type": "number"},
                                        "imag": {"type": "number"}
                                    },
                                    "required": ["real", "imag"]
                                }
                            ]
                        }
                    }
                },
                "required": ["values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "impedance_from_voltage_current",
            "description": "Compute impedance Z = V / I, inputs may be real or complex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "V": {
                        "description": "Voltage, may be real or complex",
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    },
                    "I": {
                        "description": "Current, may be real or complex",
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    }
                },
                "required": ["V", "I"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "impedance_series",
            "description": "Compute total impedance of elements in series by summing complex impedances.",
            "parameters": {
                "type": "object",
                "properties": {
                    "impedances": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "real": {"type": "number"},
                                        "imag": {"type": "number"}
                                    },
                                    "required": ["real", "imag"]
                                }
                            ]
                        }
                    }
                },
                "required": ["impedances"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "impedance_parallel",
            "description": "Compute total impedance of elements in parallel using the parallel impedance rule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "impedances": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "real": {"type": "number"},
                                        "imag": {"type": "number"}
                                    },
                                    "required": ["real", "imag"]
                                }
                            ]
                        }
                    }
                },
                "required": ["impedances"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complex_add",
            "description": "Add a list of complex or real values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "real": {"type": "number"},
                                        "imag": {"type": "number"}
                                    },
                                    "required": ["real", "imag"]
                                }
                            ]
                        }
                    }
                },
                "required": ["values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complex_multiply",
            "description": "Multiply two complex or real values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    },
                    "b": {
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    }
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complex_divide",
            "description": "Divide two complex or real values, returns a / b.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    },
                    "b": {
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    }
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "magnitude",
            "description": "Return magnitude of a real or complex value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    }
                },
                "required": ["x"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "phase_deg",
            "description": "Return phase of a complex value in degrees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    }
                },
                "required": ["x"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complex_from_polar",
            "description": "Build a complex number from magnitude and phase in degrees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "magnitude_value": {
                        "type": "number",
                        "description": "Magnitude of the complex value"
                    },
                    "phase_deg_value": {
                        "type": "number",
                        "description": "Phase in degrees"
                    }
                },
                "required": ["magnitude_value", "phase_deg_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "equation_verification",
            "description": "Verify whether the left hand side and right hand side of an equation are equal within a given tolerance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lhs": {
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    },
                    "rhs": {
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {
                                    "real": {"type": "number"},
                                    "imag": {"type": "number"}
                                },
                                "required": ["real", "imag"]
                            }
                        ]
                    },
                    "tol": {
                        "type": "number",
                        "description": "Numeric tolerance for the difference magnitude",
                        "default": 1e-6
                    }
                },
                "required": ["lhs", "rhs"]
            }
        }
    }
]


# Implementations

def ohms_law_voltage(R, I):
    return {"V": float(R * I)}

def ohms_law_current(V, R):
    return {"I": float(V / R)}

def ohms_law_resistance(V, I):
    return {"R": float(V / I)}

def voltage_add(values):
    total = sum(_complex_from_any(v) for v in values)
    return _complex_to_dict(total)

def voltage_difference(v1, v2):
    vh = _complex_from_any(v1)
    vl = _complex_from_any(v2)
    diff = vh - vl
    return _complex_to_dict(diff)

def current_add(values):
    total = sum(_complex_from_any(v) for v in values)
    return _complex_to_dict(total)

def impedance_from_voltage_current(V, I):
    Vc = _complex_from_any(V)
    Ic = _complex_from_any(I)
    Z = Vc / Ic
    return _complex_to_dict(Z)

def impedance_series(impedances):
    total = sum(_complex_from_any(z) for z in impedances)
    return _complex_to_dict(total)

def impedance_parallel(impedances):
    zs = [ _complex_from_any(z) for z in impedances ]
    if not zs:
        return _complex_to_dict(0.0)
    inv_sum = sum(1.0 / z for z in zs)
    Z_eq = 1.0 / inv_sum
    return _complex_to_dict(Z_eq)

def complex_add(values):
    total = sum(_complex_from_any(v) for v in values)
    return _complex_to_dict(total)

def complex_multiply(a, b):
    za = _complex_from_any(a)
    zb = _complex_from_any(b)
    return _complex_to_dict(za * zb)

def complex_divide(a, b):
    za = _complex_from_any(a)
    zb = _complex_from_any(b)
    return _complex_to_dict(za / zb)

def magnitude(x):
    zx = _complex_from_any(x)
    return {"magnitude": float(abs(zx))}

def phase_deg(x):
    zx = _complex_from_any(x)
    angle_rad = math.atan2(zx.imag, zx.real)
    angle_deg = angle_rad * 180.0 / math.pi
    return {"phase_deg": float(angle_deg)}

def complex_from_polar(magnitude_value, phase_deg_value):
    angle_rad = phase_deg_value * math.pi / 180.0
    z = magnitude_value * complex(math.cos(angle_rad), math.sin(angle_rad))
    return _complex_to_dict(z)

def equation_verification(lhs, rhs, tol=1e-6):
    """
    Verify whether the left hand side and right hand side of an equation are equal
    within a given tolerance.

    Arguments:
        lhs: value for the left hand side, can be real, complex, or a dict from previous tools
        rhs: value for the right hand side, same types as lhs
        tol: numeric tolerance for the difference magnitude

    Returns a dict with:
        - lhs: complex representation of the left side
        - rhs: complex representation of the right side
        - difference: lhs minus rhs as complex
        - difference_magnitude: absolute value of the difference
        - equal: True if difference magnitude is within tol, else False
        - tolerance: the tolerance used
    """
    z_lhs = _complex_from_any(lhs)
    z_rhs = _complex_from_any(rhs)
    diff = z_lhs - z_rhs
    diff_mag = abs(diff)

    return {
        "lhs": _complex_to_dict(z_lhs),
        "rhs": _complex_to_dict(z_rhs),
        "difference": _complex_to_dict(diff),
        "difference_magnitude": float(diff_mag),
        "equal": bool(diff_mag <= tol),
        "tolerance": float(tol),
    }



TOOL_IMPLS = {
    "ohms_law_voltage": ohms_law_voltage,
    "ohms_law_current": ohms_law_current,
    "ohms_law_resistance": ohms_law_resistance,
    "voltage_add": voltage_add,
    "voltage_difference": voltage_difference,
    "current_add": current_add,
    "impedance_from_voltage_current": impedance_from_voltage_current,
    "impedance_series": impedance_series,
    "impedance_parallel": impedance_parallel,
    "complex_add": complex_add,
    "complex_multiply": complex_multiply,
    "complex_divide": complex_divide,
    "magnitude": magnitude,
    "phase_deg": phase_deg,
    "complex_from_polar": complex_from_polar,
    "equation_verification": equation_verification,
}


class Model:
    """
    Supported models:
      - "openai": Azure OpenAI / OpenAI GPT models
      - "claude": Claude via OpenRouter API
      - "gemini": Gemini via OpenRouter API with reasoning support
      - "qwen": Qwen3-VL-8B Instruct Vision-Language model
      - "llava": FireLLaVA-13b Vision-Language model
      - "internvl": InternVL3-78B Vision-Language model via OpenRouter API
      - "nemotron": NVIDIA Nemotron Nano 12B VL via OpenRouter API with reasoning support
      - "glm": z-ai/glm-4.5v via OpenRouter API with reasoning support
    """
    
    def _update_token_usage(self, response):
        """Extract and update token usage from API response."""
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
                self.last_input_tokens = input_tokens
                self.last_output_tokens = output_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
            else:
                self.last_input_tokens = 0
                self.last_output_tokens = 0
        except Exception as e:
            print(f"Warning: Could not extract token usage: {e}")
            self.last_input_tokens = 0
            self.last_output_tokens = 0

    def __init__(self, name):
        self.name = name
        self.client = None
        self.model = None
        self.processor = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Token usage tracking
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if name == "openai":
            # Primary: Azure OpenAI
            self.client = AzureOpenAI(
                api_version="2024-10-21",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                organization=os.getenv("AZURE_OPENAI_ORG"),
            )
            # Fallback: OpenRouter
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

        elif name == "claude":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

        elif name == "gemini":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
        
        elif name == "qwen32b":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

        elif name == "qwen":
            repo = "Qwen/Qwen3-VL-8B-Instruct"
            self.model, self.processor = self._init_qwen(repo)

        elif name == "llava":
            repo = "llava-hf/llava-1.5-13b-hf"
            self.model, self.processor = self._init_llava(repo)

        elif name == "llama":
            repo = "meta-llama/Llama-3.2-11B-Vision" 
            self.model, self.processor = self._init_llama(repo)

        elif name == "internvl":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

        elif name == "nemotron":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

        elif name == "glm":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

        elif name == "gpt4o":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

        else:
            raise ValueError(f"Unsupported model name: {name}. Choose 'openai', 'claude', 'gemini', 'qwen', 'llava', 'internvl', 'nemotron', 'glm', or 'gpt4o'")

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    def _init_llava(self, repo_id):
            processor = AutoProcessor.from_pretrained(repo_id)
            model = LlavaForConditionalGeneration.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                device_map="auto",
            ).eval()
            return model, processor

    def _init_llama(self, repo_id):

        # from transformers import AutoProcessor, AutoModelForVision2Seq

        # processor = AutoProcessor.from_pretrained(repo_id)
        # model = AutoModelForVision2Seq.from_pretrained(repo_id)
        # print(f"Loaded LLaMA Vision {repo_id}")
        # return model, processor
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(repo_id)
        model = MllamaForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.tie_weights()
        return model, processor

    

    def _init_qwen(self, repo_id: str):
        """
        Initialize Qwen3 VL using Hugging Face Transformers.
        """
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(repo_id)

        # Fast path
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            repo_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model.eval()

        print(f"Loaded Qwen3 VL {repo_id}")
        return model, processor


    def _init_internvl(self, repo_id):
        """
        Initialize InternVL Vision-Language Model.
        Tries auto dtype first. If OOM occurs and bitsandbytes is available, tries 4-bit load.
        """
        try:
            print(f"Loading InternVL {repo_id} with auto dtype...")
            model = AutoModel.from_pretrained(
                repo_id,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
            print("Loaded with auto dtype")
            return model, processor
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and BNB_AVAILABLE:
                print("OOM with auto dtype. Retrying in 4-bit...")
                torch.cuda.empty_cache()
                gc.collect()
                model = AutoModel.from_pretrained(
                    repo_id,
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    trust_remote_code=True,
                    device_map="auto"
                )
                processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
                print("Loaded in 4-bit")
                return model, processor
            raise


    def predict(self, system_prompt, user_prompt="", reasoning=True, max_tokens=2048):
        """Text-only inference."""
        if self.name == "openai":
            variant_azure = "gpt-5.1"
            variant_openrouter = "openai/gpt-4o-2024-11-20"
            
            try:
                completion = self.client.chat.completions.create(
                    model=variant_azure,
                    temperature=0.0,
                    store=True,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                self._update_token_usage(completion)
                output = str(completion.choices[0].message.content)
                return output
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                    print(f"Azure OpenAI failed ({str(e)[:100]}), using OpenRouter fallback...")
                    completion = self.openrouter_client.chat.completions.create(
                        model=variant_openrouter,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    self._update_token_usage(completion)
                    output = str(completion.choices[0].message.content)
                    return output
                else:
                    raise
        
        elif self.name == "claude":
            variant = "anthropic/claude-sonnet-4"
            
            completion = self.client.chat.completions.create(
                model=variant,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            self._update_token_usage(completion)
            output = str(completion.choices[0].message.content)
            return output
        
        elif self.name == "gemini":
            variant = "google/gemini-2.5-flash-preview-09-2025"
            
            completion = self.client.chat.completions.create(
                model=variant,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                extra_body={"reasoning": {"enabled": True}}
            )
            self._update_token_usage(completion)
            output = str(completion.choices[0].message.content)
            return output
        
        elif self.name == "qwen32b":
            variant = "qwen/qwen3-vl-32b-instruct"
            
            completion = self.client.chat.completions.create(
                model=variant,
                temperature=0.2,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            self._update_token_usage(completion)
            output = str(completion.choices[0].message.content)
            return output

        elif self.name == "qwen":
            import torch

            # optional global speed knobs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            if self.model is None:
                raise RuntimeError("Qwen model not initialized.")

            self.model.eval()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ]

            # If your processor supports tokenize=True, this avoids extra string handling
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                padding=False,               # faster for single item
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device, non_blocking=True) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,       # lower if possible
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    temperature=0.0,
                    pad_token_id=getattr(self.processor, "pad_token_id", None) or getattr(self.model.config, "pad_token_id", None),
                    eos_token_id=getattr(self.processor, "eos_token_id", None) or getattr(self.model.config, "eos_token_id", None),
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return output.strip()
        

        if self.name == "llama":
            import torch

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

            if self.model is None:
                raise RuntimeError("LLaMA model not initialized.")

            self.model.eval()

            sys = system_prompt or ""
            usr = user_prompt or ""

            # Manual chat prompt (no apply_chat_template)
            prompt = (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n"
                f"{sys}<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{usr}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )

            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,  # important since prompt already has tokens
            ).to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,             # try 0.5 to 0.8
                    top_p=0.9,                   # try 0.85 to 0.95

                    repetition_penalty=1.12,
                    no_repeat_ngram_size=4,
                    pad_token_id=getattr(self.processor, "pad_token_id", None)
                        or getattr(self.model.config, "pad_token_id", None),
                    eos_token_id=getattr(self.processor, "eos_token_id", None)
                        or getattr(self.model.config, "eos_token_id", None),
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            output = output.replace("<|start_header_id|>", "")
            output = output.replace("<|end_header_id|>", "")
            output = output.replace("<|eot_id|>", "")
            output = output.replace("assistant", "")
            return output.strip()


        
        elif self.name == "llava":
            if self.model is None or self.processor is None:
                raise RuntimeError("LLaVA not initialized.")

            content = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

            content = content.replace("<image>", "figure").replace("</image>", "figure")
            

            inputs = self.processor(text=content, return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return text.strip()
        
        elif self.name == "internvl":
            if self.client is None:
                raise RuntimeError("InternVL client not initialized.")
            
            variant = "opengvlab/internvl3-78b"
            
            completion = self.client.chat.completions.create(
                model=variant,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            self._update_token_usage(completion)
            output = str(completion.choices[0].message.content)
            return output
        
        elif self.name == "nemotron":
            if self.client is None:
                raise RuntimeError("Nemotron client not initialized.")
            
            variant = "nvidia/nemotron-nano-12b-v2-vl:free"
            
            completion = self.client.chat.completions.create(
                model=variant,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                extra_body={"reasoning": {"enabled": True}}
            )
            self._update_token_usage(completion)
            output = str(completion.choices[0].message.content)
            return output
        
        elif self.name == "glm":
            if self.client is None:
                raise RuntimeError("GLM client not initialized.")
            
            variant = "z-ai/glm-4.5v"
            
            completion = self.client.chat.completions.create(
                model=variant,
                temperature=0.2,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                extra_body={"reasoning": {"enabled": reasoning}}
            )
            self._update_token_usage(completion)
            output = str(completion.choices[0].message.content)
            return output

        elif self.name == "gpt4o":
            variant = "openai/gpt-4o-2024-11-20"

            completion = self.client.chat.completions.create(
                model=variant,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            self._update_token_usage(completion)
            output = str(completion.choices[0].message.content)
            return output

        else:
            raise ValueError(f"predict() not supported for {self.name}")

    def predict_tool(self, system_prompt, user_prompt, images=None):
        """Text-only or multimodal inference with tool calling."""
        variant_azure = "gpt-5.1"
        variant_openrouter = "openai/gpt-4o-2024-11-20"
        
        if self.name == "openai":
            # If images provided, use vision-capable path
            if images is not None:
                # For OpenAI, use the native vision + tools API
                return self._openai_vision_with_tools_with_fallback(variant_azure, variant_openrouter, system_prompt, user_prompt, images)
            else:
                # Text-only tool calling with fallback
                try:
                    answer = self.run_with_tools(
                        client=self.client,
                        model=variant_azure,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                    return str(answer)
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                        print(f"Azure OpenAI failed ({str(e)[:100]}), using OpenRouter fallback...")
                        answer = self.run_with_tools(
                            client=self.openrouter_client,
                            model=variant_openrouter,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                        )
                        return str(answer)
                    else:
                        raise

        elif self.name == "claude":
            variant = "anthropic/claude-sonnet-4"
            # If images provided, use vision-capable path
            if images is not None:
                # For Claude, use the OpenAI-compatible vision + tools API
                return self._openai_vision_with_tools(variant, system_prompt, user_prompt, images)
            else:
                # Text-only tool calling
                answer = self.run_with_tools(
                    client=self.client,
                    model=variant,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                return str(answer)
            
        elif self.name == "qwen32b":
            variant = "qwen/qwen3-vl-32b-instruct"
            # If images provided, use vision-capable path
            if images is not None:
                return self.predict_vl(system_prompt, user_prompt, images=images)
            else:
                return self.predict(system_prompt, user_prompt)

        elif self.name == "gemini":
            variant = "google/gemini-3-flash-preview"
            # If images provided, use vision-capable path
            if images is not None:
                # For Gemini, use the OpenAI-compatible vision + tools API
                return self._openai_vision_with_tools(variant, system_prompt, user_prompt, images)
            else:
                # Text-only tool calling
                answer = self.run_with_tools(
                    client=self.client,
                    model=variant,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                return str(answer)

        elif self.name == "qwen":
            # Qwen via HuggingFace doesn't have native tool calling
            # Use predict_vl and include tool instructions in the prompt
            if images is not None:
                return self._manual_tool_orchestration_vision(variant, system_prompt, user_prompt, images)
            else:
                return self._manual_tool_orchestration_text(variant, system_prompt, user_prompt)
            
        elif self.name == "llama":
            # Qwen via HuggingFace doesn't have native tool calling
            # Use predict_vl and include tool instructions in the prompt
            if images is not None:
                tool_prompt = f"{user_prompt}\n\nYou have access to the following tools:\n{json.dumps(TOOLS, indent=2)}"
                return self.predict_vl(system_prompt, tool_prompt, images=images)
            else:
                tool_prompt = f"{user_prompt}\n\nYou have access to the following tools:\n{json.dumps(TOOLS, indent=2)}"
                return self.predict(system_prompt, tool_prompt)


        elif self.name == "llava":
            # If images provided, use vision path
            if images is not None:
                # For LLaVA, we can't easily mix vision + tool calling in the current setup
                # So we'll use predict_vl and include tool instructions in the prompt
                tool_prompt = f"{user_prompt}\n\nYou have access to the following tools:\n{json.dumps(TOOLS, indent=2)}"
                return self.predict_vl(system_prompt, tool_prompt, images=images)
            else:
                # Text-only tool calling - use simple predict since llava doesn't have native tool calling
                tool_prompt = f"{user_prompt}\n\nYou have access to the following tools:\n{json.dumps(TOOLS, indent=2)}"
                return self.predict(system_prompt, tool_prompt)

        elif self.name == "internvl":
            variant = "opengvlab/internvl3-78b"
            # InternVL doesn't support native tool calling, use manual orchestration
            if images is not None:
                return self._manual_tool_orchestration_vision(variant, system_prompt, user_prompt, images)
            else:
                return self._manual_tool_orchestration_text(variant, system_prompt, user_prompt)

        elif self.name == "nemotron":
            variant = "nvidia/nemotron-nano-12b-v2-vl:free"
            # If images provided, use vision-capable path
            if images is not None:
                # For Nemotron, use the OpenAI-compatible vision + tools API
                return self._openai_vision_with_tools(variant, system_prompt, user_prompt, images)
            else:
                # Text-only tool calling
                answer = self.run_with_tools(
                    client=self.client,
                    model=variant,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                return str(answer)
        
        elif self.name == "glm":
            variant = "z-ai/glm-4.5v"
            # If images provided, use vision-capable path
            if images is not None:
                # For GLM, use the OpenAI-compatible vision + tools API
                return self._openai_vision_with_tools(variant, system_prompt, user_prompt, images)
            else:
                # Text-only tool calling
                answer = self.run_with_tools(
                    client=self.client,
                    model=variant,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                return str(answer)

        elif self.name == "gpt4o":
            variant = "openai/gpt-4o-2024-11-20"
            if images is not None:
                return self._openai_vision_with_tools(variant, system_prompt, user_prompt, images)
            else:
                answer = self.run_with_tools(
                    client=self.client,
                    model=variant,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                return str(answer)

        else:
            raise ValueError(f"Tool calling not supported for {self.name}")

    def _build_tool_prompt_instructions(self):
        """Build tool usage instructions for models without native tool support."""
        tool_descriptions = []
        for tool in TOOLS:
            func = tool["function"]
            name = func["name"]
            desc = func["description"]
            params = func["parameters"]["properties"]
            required = func["parameters"].get("required", [])
            
            param_strs = []
            for pname, pinfo in params.items():
                req_marker = "(required)" if pname in required else "(optional)"
                pdesc = pinfo.get("description", "")
                param_strs.append(f"    - {pname} {req_marker}: {pdesc}")
            
            tool_descriptions.append(f"- {name}: {desc}\n  Parameters:\n" + "\n".join(param_strs))
        
        tools_text = "\n\n".join(tool_descriptions)
        
        instructions = f"""You have access to the following tools to help solve problems:

{tools_text}

To use a tool, respond with a JSON object in this exact format:
```json
{{"tool_call": {{"name": "<tool_name>", "arguments": {{<arguments>}}}}}}
```

After receiving the tool result, continue your reasoning and either call another tool or provide your final answer.

When you have the final answer, respond with:
```json
{{"final_answer": "<your answer>"}}
```

Important: Always use the tools for calculations rather than computing manually."""
        return instructions

    def _parse_tool_call(self, response_text):
        """Parse a tool call from the model's response text. Raises error if format is invalid."""
        import re
        
        # Try to find JSON block with tool_call
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, response_text)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                if "tool_call" in data:
                    # Validate tool_call structure
                    tool_call = data["tool_call"]
                    if not isinstance(tool_call, dict) or "name" not in tool_call:
                        raise ValueError(f"Invalid tool_call structure: missing 'name' field. Got: {tool_call}")
                    if "arguments" not in tool_call:
                        tool_call["arguments"] = {}
                    return {"type": "tool_call", "data": tool_call}
                if "final_answer" in data:
                    return {"type": "final_answer", "data": data["final_answer"]}
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON in response: {e}\nJSON block: {match[:200]}")
        
        # Try to find inline JSON
        # Look for {"tool_call": ...} pattern - more permissive regex for nested objects
        tool_pattern = r'\{\s*"tool_call"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\s*\}'
        match = re.search(tool_pattern, response_text)
        if match:
            try:
                full_match = match.group(0)
                data = json.loads(full_match)
                tool_call = data["tool_call"]
                if not isinstance(tool_call, dict) or "name" not in tool_call:
                    raise ValueError(f"Invalid tool_call structure: missing 'name' field. Got: {tool_call}")
                if "arguments" not in tool_call:
                    tool_call["arguments"] = {}
                return {"type": "tool_call", "data": tool_call}
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse inline tool_call JSON: {e}\nMatch: {match.group(0)[:200]}")
        
        # Look for {"final_answer": ...} pattern
        answer_pattern = r'\{\s*"final_answer"\s*:\s*"([^"]*)"\s*\}'
        match = re.search(answer_pattern, response_text)
        if match:
            try:
                data = json.loads(match.group(0))
                return {"type": "final_answer", "data": data["final_answer"]}
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse inline final_answer JSON: {e}\nMatch: {match.group(0)[:200]}")
        
        # No structured response found - raise error
        raise ValueError(
            f"Model response does not contain valid tool call or final answer JSON format.\n"
            f"Expected format: ```json\\n{{\"tool_call\": {{\"name\": \"...\", \"arguments\": {{...}}}}}}\\n``` "
            f"or ```json\\n{{\"final_answer\": \"...\"}}\\n```\n"
            f"Response (first 500 chars): {response_text[:500]}"
        )

    def _manual_tool_orchestration_text(self, model_variant, system_prompt, user_prompt, max_iterations=1):
        """Manual tool orchestration for text-only models without native tool support."""
        tool_instructions = self._build_tool_prompt_instructions()
        
        enhanced_system = f"{system_prompt}\n\n{tool_instructions}"
        
        messages = [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": user_prompt},
        ]
        
        for iteration in range(max_iterations):
            completion = self.client.chat.completions.create(
                model=model_variant,
                messages=messages,
                temperature=0.2,
            )
            response_text = str(completion.choices[0].message.content or "")
            
            parsed = self._parse_tool_call(response_text)
            
            if parsed["type"] == "final_answer":
                return parsed["data"]
            
            if parsed["type"] == "tool_call":
                tool_name = parsed["data"].get("name")
                tool_args = parsed["data"].get("arguments", {})
                
                # Execute the tool
                if tool_name in TOOL_IMPLS:
                    try:
                        result = TOOL_IMPLS[tool_name](**tool_args)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}
                
                # Add assistant message and tool result to conversation
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user", 
                    "content": f"Tool result for {tool_name}:\n```json\n{json.dumps(result, indent=2)}\n```\n\nContinue with your analysis. Use more tools if needed, or provide your final answer."
                })
        
        # Max iterations reached - return last response
        return response_text

    def _manual_tool_orchestration_vision(self, model_variant, system_prompt, user_prompt, images, max_iterations=1):
        """Manual tool orchestration for vision models without native tool support."""
        pil_list = _ensure_pil_list(images)
        tool_instructions = self._build_tool_prompt_instructions()
        
        enhanced_system = f"{system_prompt}\n\n{tool_instructions}"
        
        # Build initial content blocks with images
        content_blocks = []
        for im in pil_list:
            data_url = _img_to_data_url(im, fmt="PNG")
            content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        content_blocks.append({"type": "text", "text": user_prompt})
        
        messages = [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": content_blocks},
        ]
        
        for iteration in range(max_iterations):
            completion = self.client.chat.completions.create(
                model=model_variant,
                messages=messages,
                temperature=0.2,
            )
            response_text = str(completion.choices[0].message.content or "")
            
            parsed = self._parse_tool_call(response_text)
            
            if parsed["type"] == "final_answer":
                return parsed["data"]
            
            if parsed["type"] == "tool_call":
                tool_name = parsed["data"].get("name")
                tool_args = parsed["data"].get("arguments", {})
                
                # Execute the tool
                if tool_name in TOOL_IMPLS:
                    try:
                        result = TOOL_IMPLS[tool_name](**tool_args)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}
                
                # Add assistant message and tool result to conversation
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user", 
                    "content": f"Tool result for {tool_name}:\n```json\n{json.dumps(result, indent=2)}\n```\n\nContinue with your analysis. Use more tools if needed, or provide your final answer."
                })
        
        # Max iterations reached - return last response
        return response_text

    def _openai_vision_with_tools(self, model_variant, system_prompt, user_prompt, images):
        """OpenAI vision + tools path."""
        pil_list = _ensure_pil_list(images)
        
        # Build content blocks with images
        content_blocks = []
        for im in pil_list:
            data_url = _img_to_data_url(im, fmt="PNG")
            content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        content_blocks.append({"type": "text", "text": user_prompt})
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_blocks},
        ]

        while True:
            completion = self.client.chat.completions.create(
                model=model_variant,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                store=True,
            )
            self._update_token_usage(completion)
            msg = completion.choices[0].message

            # no tool calls means we are done
            if not getattr(msg, "tool_calls", None):
                return msg.content

            # add assistant message that requested tools
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                }
            )

            # execute tools
            for tool_call in msg.tool_calls:
                if tool_call.type != "function":
                    continue

                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")

                if name not in TOOL_IMPLS:
                    result = {"error": f"Unknown tool {name}"}
                else:
                    try:
                        result = TOOL_IMPLS[name](**args)
                    except Exception as e:
                        result = {"error": str(e)}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )

    def _openai_vision_with_tools_with_fallback(self, variant_azure, variant_openrouter, system_prompt, user_prompt, images):
        """OpenAI vision + tools path with OpenRouter fallback."""
        try:
            return self._openai_vision_with_tools(variant_azure, system_prompt, user_prompt, images)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                print(f"Azure OpenAI failed ({str(e)[:100]}), using OpenRouter fallback...")
                pil_list = _ensure_pil_list(images)
                
                # Build content blocks with images
                content_blocks = []
                for im in pil_list:
                    data_url = _img_to_data_url(im, fmt="PNG")
                    content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
                content_blocks.append({"type": "text", "text": user_prompt})
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_blocks},
                ]

                while True:
                    completion = self.openrouter_client.chat.completions.create(
                        model=variant_openrouter,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                    self._update_token_usage(completion)
                    msg = completion.choices[0].message

                    # no tool calls means we are done
                    if not getattr(msg, "tool_calls", None):
                        return msg.content

                    # add assistant message that requested tools
                    messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": msg.tool_calls,
                        }
                    )

                    # execute tools
                    for tool_call in msg.tool_calls:
                        if tool_call.type != "function":
                            continue

                        name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments or "{}")

                        if name not in TOOL_IMPLS:
                            result = {"error": f"Unknown tool {name}"}
                        else:
                            try:
                                result = TOOL_IMPLS[name](**args)
                            except Exception as e:
                                result = {"error": str(e)}

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": name,
                                "content": json.dumps(result),
                            }
                        )
            else:
                raise


    @torch.inference_mode()
    def predict_vl(self, system_prompt, user_prompt, images=None, max_tokens=2048, reasoning=True, variant=None):
        """
        Multimodal generation with vision support.
        Supports both OpenAI and Qwen3-VL-8B-Instruct.
        """
        pil_list = _ensure_pil_list(images)
        num_images = len(pil_list)

        # OpenAI path
        if self.name == "openai":
            if self.client is None:
                raise RuntimeError("OpenAI client is not initialized.")
            
            variant_azure = "gpt-5.1"
            variant_openrouter = "openai/gpt-4o-2024-11-20"
            
            if variant:
                variant_azure = variant
            
            content_blocks = []
            for im in pil_list:
                data_url = _img_to_data_url(im, fmt="PNG")
                content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            content_blocks.append({"type": "text", "text": user_prompt})
            
            sys_msg = {"role": "system", "content": system_prompt or ""}
            user_msg = {"role": "user", "content": content_blocks}
            
            try:
                completion = self.client.chat.completions.create(
                    model=variant_azure,
                    messages=[sys_msg, user_msg]
                )
                self._update_token_usage(completion)
                msg = completion.choices[0].message
                if isinstance(msg.content, list):
                    out_text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in msg.content
                    )
                else:
                    out_text = str(msg.content or "")
                return out_text.strip()
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                    print(f"Azure OpenAI failed ({str(e)[:100]}), using OpenRouter fallback...")
                    completion = self.openrouter_client.chat.completions.create(
                        model=variant_openrouter,
                        messages=[sys_msg, user_msg]
                    )
                    self._update_token_usage(completion)
                    msg = completion.choices[0].message
                    if isinstance(msg.content, list):
                        out_text = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in msg.content
                        )
                    else:
                        out_text = str(msg.content or "")
                    return out_text.strip()
                else:
                    raise

        # Claude path (via OpenRouter)
        if self.name == "claude":
            if self.client is None:
                raise RuntimeError("Claude client is not initialized.")
            
            if not variant:
                variant = "anthropic/claude-sonnet-4"
            
            content_blocks = []
            for im in pil_list:
                data_url = _img_to_data_url(im, fmt="PNG")
                content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            content_blocks.append({"type": "text", "text": user_prompt})
            
            sys_msg = {"role": "system", "content": system_prompt or ""}
            user_msg = {"role": "user", "content": content_blocks}
            
            completion = self.client.chat.completions.create(
                model=variant,
                messages=[sys_msg, user_msg]
            )
            self._update_token_usage(completion)
            msg = completion.choices[0].message
            if isinstance(msg.content, list):
                out_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in msg.content
                )
            else:
                out_text = str(msg.content or "")
            return out_text.strip()

        if self.name == "qwen32b":
            if self.client is None:
                raise RuntimeError("Claude client is not initialized.")
            
            if not variant:
                variant = "qwen/qwen3-vl-32b-instruct"
            
            content_blocks = []
            for im in pil_list:
                data_url = _img_to_data_url(im, fmt="PNG")
                content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            content_blocks.append({"type": "text", "text": user_prompt})
            
            sys_msg = {"role": "system", "content": system_prompt or ""}
            user_msg = {"role": "user", "content": content_blocks}
            
            completion = self.client.chat.completions.create(
                model=variant,
                max_tokens=max_tokens,
                messages=[sys_msg, user_msg]
            )
            self._update_token_usage(completion)
            msg = completion.choices[0].message
            if isinstance(msg.content, list):
                out_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in msg.content
                )
            else:
                out_text = str(msg.content or "")
            return out_text.strip()

        # Gemini path (via OpenRouter)
        if self.name == "gemini":
            if self.client is None:
                raise RuntimeError("Gemini client is not initialized.")
            
            if not variant:
                variant = "google/gemini-3-flash-preview"
            
            content_blocks = []
            for im in pil_list:
                data_url = _img_to_data_url(im, fmt="PNG")
                content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            content_blocks.append({"type": "text", "text": user_prompt})
            
            sys_msg = {"role": "system", "content": system_prompt or ""}
            user_msg = {"role": "user", "content": content_blocks}
            
            completion = self.client.chat.completions.create(
                model=variant,
                messages=[sys_msg, user_msg],
                extra_body={"reasoning": {"enabled": True}}
            )
            self._update_token_usage(completion)
            msg = completion.choices[0].message
            if isinstance(msg.content, list):
                out_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in msg.content
                )
            else:
                out_text = str(msg.content or "")
            return out_text.strip()

        # Qwen path (via HuggingFace)
        if self.name == "llama":
            import torch

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

            if self.model is None:
                raise RuntimeError("LLaMA model not initialized.")

            self.model.eval()

            # Build a manual "chat" prompt (no apply_chat_template)
            # One <|image|> token per image, then the user text.
            system_prompt = system_prompt.replace("<image>","diagram")
            user_prompt = user_prompt.replace("<image>","diagram")
            sys = system_prompt or ""
            usr = user_prompt or ""

            image_tokens = ""
            if pil_list:
                image_tokens = "\n".join(["<|image|>" for _ in pil_list]) + "\n"

            prompt = (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n"
                f"{sys}<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{image_tokens}{usr}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )

            # Processor call shaped like your Qwen call
            inputs = self.processor(
                text=[prompt],
                images=pil_list if pil_list else None,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,  # critical: we already inserted special tokens
            ).to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    temperature=0.0,
                    pad_token_id=getattr(self.processor, "pad_token_id", None)
                        or getattr(self.model.config, "pad_token_id", None),
                    eos_token_id=getattr(self.processor, "eos_token_id", None)
                        or getattr(self.model.config, "eos_token_id", None),
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return output.strip()

        # LLaVA path
        # LLaVA path
        if self.name == "llava":
            if self.model is None or self.processor is None:
                raise RuntimeError("LLaVA not initialized.")

            content = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            if num_images <= 0:
                raise ValueError("LLaVA requires an image for inference.")
            if num_images != 1:
                raise ValueError(f"LLaVA path supports exactly 1 image, got {num_images}")

            image = pil_list[0]
            prompt = f"<image>\n{content}"

            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return text.strip()

        if self.name == "qwen":
            import torch

            # optional global speed knobs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            if self.model is None:
                raise RuntimeError("Qwen model not initialized.")

            self.model.eval()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ]

            # If your processor supports tokenize=True, this avoids extra string handling
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                padding=False,               # faster for single item
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device, non_blocking=True) for k, v in inputs.items()}

            with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,       # lower if possible
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    temperature=0.0,
                    pad_token_id=getattr(self.processor, "pad_token_id", None) or getattr(self.model.config, "pad_token_id", None),
                    eos_token_id=getattr(self.processor, "eos_token_id", None) or getattr(self.model.config, "eos_token_id", None),
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return output.strip()
        
        # InternVL path (via OpenRouter)
        if self.name == "internvl":
            if self.client is None:
                raise RuntimeError("InternVL client is not initialized.")
            
            if not variant:
                variant = "opengvlab/internvl3-78b"
            
            content_blocks = []
            for im in pil_list:
                data_url = _img_to_data_url(im, fmt="PNG")
                content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            content_blocks.append({"type": "text", "text": user_prompt})
            
            sys_msg = {"role": "system", "content": system_prompt or ""}
            user_msg = {"role": "user", "content": content_blocks}
            
            completion = self.client.chat.completions.create(
                model=variant,
                messages=[sys_msg, user_msg]
            )
            self._update_token_usage(completion)
            msg = completion.choices[0].message
            if isinstance(msg.content, list):
                out_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in msg.content
                )
            else:
                out_text = str(msg.content or "")
            return out_text.strip()

        # Nemotron path (via OpenRouter with reasoning)
        if self.name == "nemotron":
            if self.client is None:
                raise RuntimeError("Nemotron client is not initialized.")
            
            if not variant:
                variant = "nvidia/nemotron-nano-12b-v2-vl:free"
            
            content_blocks = []
            for im in pil_list:
                data_url = _img_to_data_url(im, fmt="PNG")
                content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            content_blocks.append({"type": "text", "text": user_prompt})
            
            sys_msg = {"role": "system", "content": system_prompt or ""}
            user_msg = {"role": "user", "content": content_blocks}
            
            completion = self.client.chat.completions.create(
                model=variant,
                messages=[sys_msg, user_msg],
                extra_body={"reasoning": {"enabled": True}}
            )
            self._update_token_usage(completion)
            msg = completion.choices[0].message
            if isinstance(msg.content, list):
                out_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in msg.content
                )
            else:
                out_text = str(msg.content or "")
            return out_text.strip()
        
        # GLM path (via OpenRouter with reasoning)
        if self.name == "glm":
            if self.client is None:
                raise RuntimeError("GLM client is not initialized.")
            
            if not variant:
                variant = "z-ai/glm-4.5v"
            
            content_blocks = []
            for im in pil_list:
                data_url = _img_to_data_url(im, fmt="PNG")
                content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            content_blocks.append({"type": "text", "text": user_prompt})
            
            sys_msg = {"role": "system", "content": system_prompt or ""}
            user_msg = {"role": "user", "content": content_blocks}
            
            completion = self.client.chat.completions.create(
                model=variant,
                max_tokens=max_tokens,
                messages=[sys_msg, user_msg],
                extra_body={"reasoning": {"enabled": reasoning}}
            )
            self._update_token_usage(completion)
            msg = completion.choices[0].message
            if isinstance(msg.content, list):
                out_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in msg.content
                )
            else:
                out_text = str(msg.content or "")
            return out_text.strip()

        if self.name == "gpt4o":
            if self.client is None:
                raise RuntimeError("GPT-4o client is not initialized.")

            if not variant:
                variant = "openai/gpt-4o-2024-11-20"

            content_blocks = []
            for im in pil_list:
                data_url = _img_to_data_url(im, fmt="PNG")
                content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            content_blocks.append({"type": "text", "text": user_prompt})

            sys_msg = {"role": "system", "content": system_prompt or ""}
            user_msg = {"role": "user", "content": content_blocks}

            completion = self.client.chat.completions.create(
                model=variant,
                messages=[sys_msg, user_msg]
            )
            self._update_token_usage(completion)
            msg = completion.choices[0].message
            if isinstance(msg.content, list):
                out_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in msg.content
                )
            else:
                out_text = str(msg.content or "")
            return out_text.strip()

        raise ValueError(f"predict_vl not supported for {self.name}")

    def run_with_tools(self, client, model, system_prompt, user_prompt):

        if "gpt" in model.lower() or "openai" in model.lower() or "claude" in model.lower() or "anthropic" in model.lower() or "gemini" in model.lower() or "google" in model.lower() or "qwen" in model.lower() or "internvl" in model.lower() or "opengvlab" in model.lower() or "nemotron" in model.lower() or "nvidia" in model.lower() or "glm" in model.lower() or "z-ai" in model.lower():
            # ******** GPT / OpenAI / Claude / Gemini / Qwen / InternVL / Nemotron (OpenRouter) native tools path ********
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            while True:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    store=True,
                )
                self._update_token_usage(completion)
                msg = completion.choices[0].message

                # no tool calls means we are done
                if not getattr(msg, "tool_calls", None):
                    return msg.content

                # add assistant message that requested tools
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": msg.tool_calls,
                    }
                )

                # execute tools
                for tool_call in msg.tool_calls:
                    if tool_call.type != "function":
                        continue

                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments or "{}")

                    if name not in TOOL_IMPLS:
                        result = {"error": f"Unknown tool {name}"}
                    else:
                        try:
                            result = TOOL_IMPLS[name](**args)
                        except Exception as e:
                            result = {"error": str(e)}

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )

            # end GPT path

        else:
            raise ValueError(f"run_with_tools does not support model: {model}")
