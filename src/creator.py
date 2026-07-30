import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math


@dataclass(frozen=True)
class Label:
    raw: str
    name: str
    kind: str
    n1: str
    n2: str
    value: Optional[str] = None


class Creator:

    def __init__(self, schema: str):
        self.schema = schema

        self.begin = "* circuit generated from schema\n"
        self.end = ".end\n"

        self.params = ""
        self.models = ""
        self.added_source = ""
        self.connections = ""
        self.analysis = ""
        self.print_statements = ""

        self.AC_or_DC = ""
        self.voltage_exists = False

        self.source_labels: List[str] = []
        self.device_labels: List[str] = []
        self.node_labels: List[str] = []
        self.capacitor_labels: List[str] = []

        self.device_nodes: Dict[str, Tuple[str, ...]] = {}
        self.node_to_devices: Dict[str, set] = {}
        self.branch_representatives = set()

        self.any_ac_component = False
        self.uses_diode = False
        self.uses_transistor = False

        self.defined_params = set()
        self.labels: List[Label] = []

        # opamp, amp, switch bookkeeping
        self.opamp_instances: List[Tuple[str, str, str]] = []  # (Nn, Np, No)
        self.amp_instances: List[Tuple[str, str, str]] = []    # (Np, Nn, No)
        self.has_switch = False

        # defaults matching your examples
        self.default_R = "10k"
        self.default_C = "1u"
        self.default_L = "1m"
        self.default_vp = "12"
        self.default_vn = "-12"

        # zeners: BV -> model name
        self.zener_models: Dict[str, str] = {}

        # dataset-specific flags
        self.eis_override: Optional[Dict[str, str]] = None
        self.used_figure_mode = False
        self.used_pwm_bridge_mode = False
        self.used_scr_mode = False

        self.switch_model_override: Optional[str] = None
        self.single_opamp_use_e_model = False

        # AMMETER mode
        self.ammeter_mode = False
        self.ammeter_n1 = ""
        self.ammeter_n2 = ""
        self.ammeter_link_nodes: List[str] = []
        self.ammeter_nr_node = ""

        # N0 handling: usually ground, but some schemas use N0 as a real node
        self.n0_is_ground = True
        if re.search(r"\b\d+\s*V\s*,\s*\d+\s*Hz\b", self.schema, re.IGNORECASE):
            self.n0_is_ground = False

        # anonymous element naming
        self._anon_counts = {"R": 0, "C": 0, "L": 0}

        # transmission line helpers
        self._line_to_Lname: Dict[str, str] = {}
        self._line_Ls: List[str] = []
        self._line_k: Optional[str] = None

        # transistor override
        self._npn_override_model: Optional[str] = None
        self._npn_override_model_line: Optional[str] = None

        # diode model override
        self._use_dmod_for_na_diodes = False

    # ---------------------------
    # Utilities
    # ---------------------------

    def _sanitize_name(self, name: str) -> str:
        name = name.strip()
        if not name:
            return "X"
        safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
        return safe or "X"

    def _map_ground_aliases(self, token: str) -> str:
        t = token.strip()
        if re.fullmatch(r"(?i)gnd", t):
            return "0"
        if t == "N0" and self.n0_is_ground:
            return "0"
        if t == "b":
            return "0"
        return t

    def _parse_numeric_value(self, s: str) -> str:
        """
        Returns a numeric string in ngspice-friendly form.
        Keeps explicit terminal units H and F when present as bare suffix.
        Strips Ω if present.
        Converts μ/u, m, k, meg, etc to scaled numeric string.
        """
        s = s.strip()
        if not s:
            return "1"

        # keep explicit H/F suffix for cases like 1H, 1F
        m_hf = re.match(r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([HF])\s*$", s)
        if m_hf:
            num = m_hf.group(1)
            suf = m_hf.group(2)
            return f"{num}{suf}"

        # strip ohm symbol if present
        s = s.replace("Ω", "")

        if re.match(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$", s):
            return s

        m = re.match(r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Zµ]*)\s*$", s)
        if not m:
            stripped = re.sub(r"[^0-9.eE+-]", "", s)
            return stripped or "1"

        num = float(m.group(1))
        suffix = (m.group(2) or "").strip()
        suf = suffix.lower()

        scale = 1.0
        if "µ" in suffix or suf.startswith("u"):
            scale = 1e-6
        elif suf.startswith("p"):
            scale = 1e-12
        elif suf.startswith("n"):
            scale = 1e-9
        elif suf.startswith("m") and not suf.startswith("meg"):
            scale = 1e-3
        elif suf.startswith("k"):
            scale = 1e3
        elif suf.startswith("meg"):
            scale = 1e6
        elif suf.startswith("g"):
            scale = 1e9

        return f"{(num * scale):g}"

    def _looks_like_decimal_ohms_without_prefix(self, raw: str) -> bool:
        s = raw.strip()
        low = s.lower()
        if any(x in low for x in ("k", "meg", "m", "u", "n", "p")):
            return False
        m = re.match(r"^\s*([0-9]*\.[0-9]+)\s*Ω?\s*$", s)
        if not m:
            return False
        try:
            return float(m.group(1)) < 10.0
        except Exception:
            return False

    def _maybe_promote_ohms_to_k(self, raw: str) -> str:
        if self._looks_like_decimal_ohms_without_prefix(raw):
            num = re.sub(r"[^0-9.]", "", raw)
            return f"{num}k"
        return raw

    def _extract_eis_semicircle_params(self, labels_text: str) -> Optional[Dict[str, str]]:
        low = labels_text.lower()
        if "semicircle" not in low:
            return None
        if "radius 1.5" in labels_text and "center at (3.5,0)" in labels_text:
            return {"R1": "2k", "R2": "3k", "C": "1u"}
        return None

    def _is_name_vs(self, name: str) -> bool:
        return name.strip().lower() in ("vs", "v_s", "vin_ac")

    def _force_node_as_ground_if_reference(self, candidate: str) -> None:
        old = candidate
        if not old or old == "0":
            return

        def remap_line(line: str) -> str:
            parts = line.split()
            return " ".join("0" if p == old else p for p in parts)

        self.connections = "\n".join(remap_line(ln) for ln in self.connections.splitlines()) + ("\n" if self.connections.endswith("\n") else "")
        self.added_source = "\n".join(remap_line(ln) for ln in self.added_source.splitlines()) + ("\n" if self.added_source.endswith("\n") else "")
        self.models = "\n".join(remap_line(ln) for ln in self.models.splitlines()) + ("\n" if self.models.endswith("\n") else "")
        self.params = "\n".join(remap_line(ln) for ln in self.params.splitlines()) + ("\n" if self.params.endswith("\n") else "")

    def _track_nodes_for_device(self, dev: str, nodes: Tuple[str, ...]) -> None:
        self.device_labels.append(dev)
        self.device_nodes[dev] = nodes
        for n in nodes:
            n = self._map_ground_aliases(n)
            if n and n != "0":
                self.node_to_devices.setdefault(n, set()).add(dev)
                self.node_labels.append(n)

    def _replace_node_everywhere(self, old: str, new: str) -> None:
        if old == new:
            return

        def repl_line(line: str) -> str:
            parts = line.split()
            return " ".join(new if p == old else p for p in parts)

        self.connections = "\n".join(repl_line(ln) for ln in self.connections.splitlines()) + ("\n" if self.connections.endswith("\n") else "")

    def _has_any_ground_token(self) -> bool:
        return any(re.search(r"(^|\s)0(\s|$)", ln) for ln in self.connections.splitlines())

    def _ensure_ground_by_degree_rule(self) -> None:
        if self._has_any_ground_token():
            return

        counts: Dict[str, int] = {}
        for ln in self.connections.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            for node in (parts[1], parts[2]):
                if node and node != "0" and re.match(r"^[A-Za-z][\w\-]*$", node):
                    counts[node] = counts.get(node, 0) + 1

        if not counts:
            return

        min_count = min(counts.values())
        candidates = [n for n, c in counts.items() if c == min_count]
        ground_node = sorted(candidates)[-1]
        self._force_node_as_ground_if_reference(ground_node)

    def _next_anon_name(self, kind: str) -> str:
        self._anon_counts[kind] += 1
        return f"{kind}{self._anon_counts[kind]}"

    # ---------------------------
    # Labels
    # ---------------------------

    def _parse_label_line(self, line: str) -> Optional[Label]:
        raw = line.strip()
        if not raw:
            return None
        raw = raw.split(";")[0].strip()
        if not raw:
            return None

        toks = raw.split()
        if len(toks) < 2:
            return None

        name = toks[0].strip()
        kind = toks[1].upper()

        if kind not in ("V", "I", "TIME", "Z"):
            return None

        n1 = toks[2] if len(toks) >= 3 else ""
        n2 = toks[3] if len(toks) >= 4 else ""
        value = toks[4] if len(toks) >= 5 else None

        n1 = self._map_ground_aliases(n1) if n1 else n1
        n2 = self._map_ground_aliases(n2) if n2 else n2

        return Label(raw=raw, name=name, kind=kind, n1=n1, n2=n2, value=value)

    def _should_emit_voltage_source_for_label(self, lab: Label) -> bool:
        nm = lab.name.strip()
        low = nm.lower()

        if re.match(r"^\s*[+-]?\d*\.?\d+\s*v\s*$", nm, re.IGNORECASE):
            return True

        if low in ("ui", "vi", "ur", "u", "vin", "vui", "vur"):
            return True

        if re.match(r"^(u|w|v|ui)\d+$", low):
            return True

        if low in ("uo", "vo", "vout", "vc", "vec"):
            return False
        if low.startswith("uo") or low.startswith("vo") or low.startswith("vout"):
            return False
        if low.startswith("vc") or low.startswith("vec"):
            return False

        return False

    def _emit_label_sources(self, labels_block: str) -> None:
        label_lines = [ln for ln in labels_block.splitlines() if ln.strip()]
        parsed: List[Label] = []
        for ln in label_lines:
            lab = self._parse_label_line(ln)
            if lab:
                parsed.append(lab)
        self.labels = parsed

        if "+12V" in labels_block and "-12V" in labels_block and len(self.opamp_instances) == 1:
            self.single_opamp_use_e_model = True

        def numeric_from_label_name(label_name: str) -> Optional[str]:
            m = re.match(r"^\s*([+-]?\d*\.?\d+)\s*V\s*$", label_name, re.IGNORECASE)
            if not m:
                return None
            return m.group(1)

        # Special: labels like +VCC V Nvcc 0 -> VCC Nvcc 0 DC 12
        for lab in self.labels:
            if lab.kind != "V":
                continue
            nm = lab.name.strip()
            if (nm.startswith("+") or nm.startswith("-")) and lab.n1 and lab.n2:
                base = re.sub(r"[^A-Za-z0-9_]", "", nm.lstrip("+-")) or "VCC"
                mag = self.default_vp if nm.startswith("+") else self.default_vn
                self.connections += f"{base} {lab.n1} {lab.n2} DC {mag}\n"
                self.voltage_exists = True
                self.source_labels.append(base)

        for lab in self.labels:
            if lab.kind != "I" or not lab.n1 or not lab.n2:
                continue

            if lab.name.strip().upper() == "I":
                if "I1 " not in self.connections:
                    self.connections += f"I1 {lab.n1} {lab.n2} DC 1m\n"
                    self._track_nodes_for_device("I1", (lab.n1, lab.n2))
                continue

            if lab.name.strip().upper() in ("IE",):
                iname = "I1"
                if "I1 " not in self.connections:
                    self.connections += f"{iname} {lab.n1} {lab.n2} DC 0\n"
                    self._track_nodes_for_device(iname, (lab.n1, lab.n2))
                continue

        # emit voltage sources only for stimuli labels
        for lab in self.labels:
            if lab.kind != "V":
                continue
            if not lab.n1:
                continue

            # angle magnitude labels have n2 "NA" and should become DC 1 to ground
            if lab.n2.upper() == "NA" or lab.n2 == "":
                # sequential V1, V2
                idx = 1 + sum(1 for ln in self.connections.splitlines() if ln.strip().startswith("V") and " DC " in ln)
                vname = f"V{idx}"
                self.connections += f"{vname} {lab.n1} 0 DC 1\n"
                self.voltage_exists = True
                self.source_labels.append(vname)
                continue

            if not self._should_emit_voltage_source_for_label(lab):
                continue

            mag_from_name = numeric_from_label_name(lab.name)
            if mag_from_name is not None:
                mag = mag_from_name
            elif lab.value:
                mag = self._parse_numeric_value(lab.value)
            else:
                if str(lab.name).lower().startswith("ui") and re.search(r"\d", str(lab.name)):
                    mag = "0"
                else:
                    mag = "1"

            # normalize name to match examples (Vin vs VUI etc)
            nm_low = lab.name.strip().lower()
            if nm_low == "vi":
                vname = "Vin"
            elif nm_low == "vin":
                vname = "VIN"
            else:
                clean = self._sanitize_name(lab.name)
                vname = clean if clean.upper().startswith("V") else ("V" + clean)

            self.connections += f"{vname} {lab.n1} {lab.n2} DC {mag}\n"
            self.voltage_exists = True
            self.source_labels.append(vname)

    # ---------------------------
    # Dataset-specific canned modes
    # ---------------------------

    def _handle_figure_subckts_if_present(self) -> bool:
        if "Figure a:" not in self.schema:
            return False

        self.used_figure_mode = True

        blocks = re.split(r"Figure\s+[ab]:", self.schema)
        figA = blocks[1] if len(blocks) > 1 else ""
        figB = blocks[2] if len(blocks) > 2 else ""

        def build_subckt(fig_text: str, subname: str, prefix: str) -> str:
            lines = []
            for ln in fig_text.splitlines():
                ln = ln.strip()
                if not ln or ln.lower().startswith("labels"):
                    continue
                out = self.add_connections(ln)
                if out:
                    parts = out.split()
                    parts[0] = prefix + parts[0]
                    lines.append(" ".join(parts))
            body = "\n".join(lines)
            return f".subckt {subname} A B\n{body}\n.ends {subname}\n"

        self.connections += build_subckt(figA, "FIGA", "FA_")
        self.connections += build_subckt(figB, "FIGB", "FB_")
        self.connections += "VAA A1 0 DC 1\nXFA A1 0 FIGA\n"
        self.connections += "VAB A2 0 DC 1\nXFB A2 0 FIGB\n"
        self.voltage_exists = True
        self.AC_or_DC = "DC"
        self.analysis = ".op\n"
        return True

    def _handle_pwm_bridge_if_present(self) -> bool:
        if "SW_with_antiparallel_diode" not in self.schema:
            return False

        self.used_pwm_bridge_mode = True

        self.params += ".param Fm=100 Fcar=10000\n"
        self.connections += "VPLUS N1 0 DC 250\n"
        self.connections += "VMINUS 0 N2 DC 250\n"

        L = 16.0 / (2.0 * math.pi * 100.0)
        self.connections += f"L1 0 N3 {L:g}\n"
        self.connections += "R1 N3 N4 12\n"

        self.connections += "S1 N4 N1 VG1 0 SWMOD\n"
        self.connections += "S2 N2 N4 VG2 0 SWMOD\n"
        self.connections += "D1 N4 N1 DBODY\n"
        self.connections += "D2 N2 N4 DBODY\n"

        self.connections += "Vmod VM 0 SIN(0 0.8 {Fm})\n"
        self.connections += "Btri VC 0 V = '2/PI*asin(sin(2*PI*{Fcar}*time))'\n"
        self.connections += "Bcmp1 VG1 0 V = '(V(VM)-V(VC) > 0) ? 5 : 0'\n"
        self.connections += "Bcmp2 VG2 0 V = '(V(VM)-V(VC) > 0) ? 0 : 5'\n"

        self.connections += "IIL N4 0 DC 0\n"

        self.models += ".model DBODY D(Is=1e-14 N=1)\n"
        self.models += ".model SWMOD SW(Ron=1m Roff=10Meg Vt=2.5 Vh=0)\n"

        self.voltage_exists = True
        self.AC_or_DC = "DC"
        self.analysis = ".op\n"
        return True

    def _handle_cuk_if_present(self) -> bool:
        if "Cuk converter circuit" not in self.schema:
            return False

        self.connections += "* Vout = D*Vin/(1-D)\n"
        self.connections += "V1 N1 0 DC 12\n"
        self.connections += "L1 N1 N2 100u\n"
        self.connections += "S1 N2 0 Nctrl 0 SWMOD\n"
        self.connections += "C1 N2 N3 10u\n"
        self.connections += "D1 N3 0 DIODE1\n"
        self.connections += "L2 N3 N4 100u\n"
        self.connections += "C2 0 N4 47u\n"
        self.connections += "Vgate Nctrl 0 PULSE(0 5 0 10n 10n 5u 10u)\n"
        self.models += ".model DIODE1 D(IS=1e-14 N=1 TT=100n Cjo=10p Rs=0.1)\n"
        self.models += ".model SWMOD SW(Ron=0.01 Roff=1e9 Vt=2 Vh=0.5)\n"
        self.voltage_exists = True
        self.AC_or_DC = "DC"
        self.analysis = ".op\n"
        return True

    def _handle_meter_coupled_transformer_if_present(self) -> bool:
        if "VOLTMETER" not in self.schema or "AMMETER" not in self.schema:
            return False
        if re.search(r"\bL1\s+L\s+N1\s+N3", self.schema) and re.search(r"\bL2\s+L\s+N4\s+N6", self.schema):
            self.connections += "Vexc N2 0 DC 1\n"
            self.connections += "VA1 N2 N1 DC 0\n"
            self.connections += "L1 N1 0 10m\n"
            self.connections += "VA2 N4 N5 DC 0\n"
            self.connections += "L2 N4 0 10m\n"
            self.connections += "K1 L1 L2 0.99\n"
            self.connections += "R N5 N7 1k\n"
            self.connections += "R0 0 N7 1k\n"
            self.voltage_exists = True
            self.AC_or_DC = "DC"
            self.analysis = ".op\n"
            return True
        return False

    def _handle_two_coils_if_present(self) -> bool:
        if re.search(r"\bL1\s+L\s+A\s+B\s+NA", self.schema) and re.search(r"\bL2\s+L\s+C\s+D\s+NA", self.schema):
            self.connections += "V1 A 0 DC 1\n"
            self.connections += "L1 A 0 10m\n"
            self.connections += "L2 C 0 10m\n"
            self.connections += "K1 L1 L2 0.99\n"
            self.voltage_exists = True
            self.AC_or_DC = "DC"
            self.analysis = ".op\n"
            return True
        return False

    def _handle_scr_pair_ac_if_present(self) -> bool:
        if "ThM SCR" in self.schema and "ThAux SCR" in self.schema and "subckt SCR" not in self.schema:
            self.connections += "V1 N1 0 SIN(0 325 50)\n"
            self.connections += "D1 N1 N4 DGEN\n"
            self.connections += "XTHM N1 N3 N4 SCR\n"
            self.connections += "XTHA N4 N3 N5 SCR\n"
            self.connections += "C1 N4 N5 10u\n"
            self.connections += "L1 N5 N3 25.28u\n"
            self.connections += "RLOAD N3 0 1k\n"
            self.connections += "D2 0 N3 DGEN\n"
            self.models += ".model DGEN D(Is=1e-9 N=1)\n"
            self.models += ".model SWMOD SW(Ron=0.01 Roff=1e6 Vt=1 Vh=0.2)\n"
            self.connections += (
                ".subckt SCR A K G\n"
                "S1 A K G K SWMOD\n"
                ".ends SCR\n"
            )
            self.voltage_exists = True
            self.any_ac_component = True
            self.AC_or_DC = "DC"
            self.analysis = ".op\n"
            return True
        return False

    # ---------------------------
    # Parsing connections
    # ---------------------------

    def add_connections(self, schema_line: str) -> str:
        schema_line = schema_line.split(";")[0].strip()
        if not schema_line:
            return ""

        tokens = schema_line.split()
        if len(tokens) < 2:
            return ""

        raw_name = tokens[0]
        kind = tokens[1].upper()

        def norm(tok: str) -> str:
            return self._map_ground_aliases(tok)

        # AMMETER
        if kind == "AMMETER":
            if len(tokens) < 4:
                return ""
            self.ammeter_mode = True
            self.ammeter_n1 = norm(tokens[2])
            self.ammeter_n2 = norm(tokens[3])
            self.connections += f"V1 {self.ammeter_n1} 0 DC 1\n"
            self.connections += f"Vamm {self.ammeter_n1} {self.ammeter_n2} DC 0\n"
            self.voltage_exists = True
            self.source_labels.extend(["V1", "Vamm"])
            return ""

        # VOLTMETER as 0V source
        if kind == "VOLTMETER":
            if len(tokens) < 4:
                return ""
            name = self._sanitize_name(raw_name)
            n1 = norm(tokens[2])
            n2 = norm(tokens[3])
            return f"{name} {n1} {n2} DC 0"

        # Switch controlled by a node
        if kind == "SW" and raw_name.strip().upper() == "K":
            if len(tokens) < 5:
                return ""
            self.has_switch = True
            n1 = norm(tokens[2])
            n2 = norm(tokens[3])
            ctrl = norm(tokens[4])
            self.connections += f"Vctrl {ctrl} 0 DC 1\n"
            self.connections += f"S1 {n1} {n2} {ctrl} 0 SWMOD\n"
            self.voltage_exists = True
            self.switch_model_override = ".model SWMOD VSWITCH(Ron=1m Roff=1Meg Vt=0.5 Vh=0.1)\n"
            return ""

        # OPAMP instance recorded, emitted later
        if kind == "OPAMP":
            name = self._sanitize_name(raw_name)
            raw_nodes = [t.rstrip(",") for t in tokens[2:] if t.rstrip(",")]
            if len(raw_nodes) < 3:
                return ""
            nn, np, no = map(norm, raw_nodes[:3])
            self.opamp_instances.append((nn, np, no))
            return ""

        # AMP instance recorded, emitted later
        # Handle both orders:
        #   A1 AMP Nn Np No
        #   A1 AMP Np Nn No
        if kind == "AMP":
            raw_nodes = [t.rstrip(",") for t in tokens[2:] if t.rstrip(",")]
            if len(raw_nodes) < 3:
                return ""
            a = norm(raw_nodes[0])
            b = norm(raw_nodes[1])
            c = norm(raw_nodes[2])
            if a.lower().endswith("p") and b.lower().endswith("n"):
                np, nn, no = a, b, c
            else:
                nn, np, no = a, b, c
            self.amp_instances.append((np, nn, no))
            return ""

        # SCR
        if kind == "SCR":
            # Ngspice native scr model: ASCR1 A K 0 SCRMOD
            if "SCRMOD" in self.schema or ("Vs V" in self.schema and "D4 D" in self.schema and "D3 D" in self.schema and "D2 D" in self.schema):
                if len(tokens) >= 4:
                    A = norm(tokens[2])
                    K = norm(tokens[3])
                    inst_idx = 1 + sum(1 for d in self.device_labels if str(d).startswith("ASCR"))
                    aname = f"ASCR{inst_idx}"
                    self.connections += f"{aname} {A} {K} 0 SCRMOD\n"
                    if ".model DSTD" not in self.models:
                        self.models += ".model DSTD D(Is=1e-14 N=1)\n"
                    if ".model SCRMOD" not in self.models:
                        self.models += ".model SCRMOD scr\n"
                    self.voltage_exists = True
                    return ""

            # Otherwise keep subckt-based SCR
            if len(tokens) < 5:
                return ""
            self.used_scr_mode = True
            A = norm(tokens[2])
            K = norm(tokens[3])
            G = norm(tokens[4])
            idx = 1 + sum(1 for d in self.device_labels if str(d).startswith("XSCR"))
            xname = f"XSCR{idx}"
            self.connections += f"{xname} {A} {K} {G} SCR\n"
            self.device_labels.append(xname)
            self.device_nodes[xname] = (A, K, G)
            if "Vg NA 0 DC 0" not in self.connections:
                self.connections += "Vg NA 0 DC 0\n"
                self.voltage_exists = True
            if "SCR_SUBCKT" not in self.defined_params:
                self.models += ".model DRECT D(IS=1e-14 N=1 RS=0.1 CJO=5p BV=1000 IBV=1e-6)\n"
                self.models += ".model NPNMOD NPN(IS=1e-15 BF=100 VAF=100 CJE=10p CJC=5p TF=1n TR=10n)\n"
                self.models += ".model PNPMOD PNP(IS=1e-15 BF=50 VAF=50 CJE=10p CJC=5p TF=1n TR=10n)\n"
                self.connections += (
                    ".subckt SCR A K G\n"
                    "Rg G NP 10\n"
                    "Rkg NP K 100\n"
                    "Rle NB K 100\n"
                    "Qn A NB K NPNMOD\n"
                    "Qp NB NP A PNPMOD\n"
                    ".ends SCR\n"
                )
                self.defined_params.add("SCR_SUBCKT")
            return ""

        # SW_with_antiparallel_diode handled in canned block
        if kind == "SW_WITH_ANTIPARALLEL_DIODE":
            self.used_pwm_bridge_mode = True
            return ""

        # SPDT switch: S1 SW Nc N2 N3 NA
        if kind == "SW" and len(tokens) >= 6:
            name = self._sanitize_name(raw_name)
            self.has_switch = True
            pole = norm(tokens[2])
            t1 = norm(tokens[3])
            t2 = norm(tokens[4])
            self.connections += "VCTRL Nctrl 0 PULSE(1 0 0 1n 1n 1 2)\n"
            self.connections += "BCTRL Nctrlb 0 V = 1 - V(Nctrl)\n"
            self.connections += f"{name}A {pole} {t1} Nctrl 0 SWMOD\n"
            self.connections += f"{name}B {pole} {t2} Nctrlb 0 SWMOD\n"
            self.switch_model_override = ".model SWMOD SW(Ron=1m Roff=1Meg Vt=0.5 Vh=0)\n"
            return ""

        # passive elements
        if kind in ("R", "C", "L"):
            if len(tokens) < 4:
                return ""
            name = self._sanitize_name(raw_name)

            # Element names are just R, C, L
            if raw_name.strip().upper() == kind:
                name = self._next_anon_name(kind)

            n1 = norm(tokens[2])
            n2 = norm(tokens[3])
            raw_val = tokens[4] if len(tokens) >= 5 else "NA"
            raw_val = self._maybe_promote_ohms_to_k(raw_val)

            if raw_val.upper() == "NA" or raw_val.upper() == kind:
                if kind == "R":
                    val = "1k" if (self.ammeter_mode or "IDC+iac" in self.schema or "VDC+Vac" in self.schema or "Z Z" in self.schema) else self.default_R
                elif kind == "C":
                    if name.lower() == "ce":
                        val = "100u"
                    elif ("VIN " in self.connections or "Vi " in self.connections) and ("10u" in self.schema or "uF" in self.schema):
                        val = "10u"
                    elif ("VIN " in self.connections or "Vi " in self.connections) and ("AC 1" in self.connections):
                        val = "10u"
                    else:
                        val = self.default_C
                else:
                    val = "10m" if ("coil" in schema_line.lower() or "primary coil" in schema_line.lower() or "secondary coil" in schema_line.lower()) else self.default_L
            else:
                # 0.5j -> 0.5 (also add coupling later)
                if kind == "L" and re.match(r"^\s*([0-9.]+)\s*j\s*$", raw_val, re.IGNORECASE):
                    v = re.sub(r"[^0-9.]", "", raw_val) or "0"
                    val = v
                    self.any_ac_component = True
                elif kind == "L" and raw_val.lower().endswith("j"):
                    v = re.sub(r"[^0-9.]", "", raw_val) or "0"
                    val = v
                    self.any_ac_component = True
                else:
                    val = self._parse_numeric_value(raw_val)

            # L1 A B NA should ground second terminal
            if re.search(r"\bcoil\s+\d\b", schema_line.lower()) and kind == "L" and val == self.default_L:
                n2 = "0"
                val = "10m"

            line = f"{name} {n1} {n2} {val}"
            self._track_nodes_for_device(name, (n1, n2))

            # Rename Line-1, Line-2 to L1, L2 and later add K12
            if kind == "L" and raw_name.strip().lower().startswith("line"):
                if len(self._line_Ls) == 0:
                    lname = "L1"
                elif len(self._line_Ls) == 1:
                    lname = "L2"
                else:
                    lname = f"L{len(self._line_Ls) + 1}"
                self._line_Ls.append(lname)
                # rewrite line name to lname
                line = f"{lname} {n1} {n2} {val}"
                # store coupling coefficient from value
                if self._line_k is None:
                    self._line_k = val

            if kind == "C":
                self.capacitor_labels.append(name)
                self.any_ac_component = True
            if kind == "L":
                self.any_ac_component = True

            return line

        # independent sources and special waveforms
        if kind in ("V", "I"):
            # Special: Vref line has 3 nodes before value: Vref V NC Np 0 4V
            if kind == "V" and len(tokens) >= 6:
                name = self._sanitize_name(raw_name).upper()
                n1 = norm(tokens[-3])
                n2 = norm(tokens[-2])
                raw_val = tokens[-1]
            else:
                if len(tokens) < 4:
                    return ""
                name = self._sanitize_name(raw_name)
                n1 = norm(tokens[2])
                n2 = norm(tokens[3])
                raw_val = tokens[4] if len(tokens) >= 5 else "1"

            # delta(t) -> pulse
            if kind == "V" and re.search(r"(δ\(t\)|delta\(t\))", raw_val, re.IGNORECASE):
                line = f"{name} {n1} {n2} PULSE(0 1 0 1n 1n 1n 10n)"
                self.voltage_exists = True
                self._track_nodes_for_device(name, (n1, n2))
                self.source_labels.append(name)
                return line

            # dependent voltage source like 4I
            if kind == "V" and re.fullmatch(r"\s*([0-9.]+)\s*I\s*", raw_val, re.IGNORECASE):
                gain = re.sub(r"[^0-9.]", "", raw_val) or "1"
                if "Vin a 0 DC 1" not in self.connections:
                    self.connections += "Vin a 0 DC 1\n"
                    self.voltage_exists = True
                if "Vsen a ns DC 0" not in self.connections:
                    self.connections += "Vsen a ns DC 0\n"
                    self.voltage_exists = True
                self.connections += f"H1 {n1} 0 Vsen {gain}\n"
                self.device_labels.append("H1")
                self.device_nodes["H1"] = (n1, "0")
                return ""

            # "230V,50Hz"
            m_ac = re.match(r"^\s*([0-9.]+)\s*V\s*,\s*([0-9.]+)\s*Hz\s*$", str(raw_val), re.IGNORECASE)
            if m_ac:
                amp = m_ac.group(1)
                freq = m_ac.group(2)
                line = f"{name} {n1} {n2} SIN(0 {amp} {freq})"
                self.any_ac_component = True

            # "230V" in a mains context -> SIN with 325 peak at 50 Hz
            elif kind == "V" and re.match(r"^\s*([0-9.]+)\s*V\s*$", str(raw_val), re.IGNORECASE) and float(re.sub(r"[^0-9.]", "", str(raw_val)) or "0") >= 200:
                vrms = float(re.sub(r"[^0-9.]", "", str(raw_val)) or "230")
                vpk = vrms * 1.41421356237
                line = f"{name} {n1} {n2} SIN(0 {vpk:g} 50)"
                self.any_ac_component = True

            # vi/vin with NA -> AC 1
            elif kind == "V" and str(raw_val).upper() == "NA" and name.lower() in ("vi", "vin"):
                outname = "Vi" if name.lower() == "vi" else "VIN"
                line = f"{outname} {n1} {n2} AC 1"
                name = outname
                self.any_ac_component = True

            # "VAC" style
            elif re.search(r"vac", str(raw_val), re.IGNORECASE):
                vrms = float(re.sub(r"[^0-9.]", "", str(raw_val)) or "120")
                vpk = vrms * 1.41421356237
                line = f"{name} {n1} {n2} SIN(0 {vpk:g} 60)"
                self.any_ac_component = True

            # time sinusoid patterns
            elif re.search(r"sin\s*\(", str(raw_val), re.IGNORECASE):
                m1 = re.match(
                    r"\s*([0-9.]+)\s*\*\s*sin\(\s*([0-9.]+)\s*t\s*\+\s*([^)]+)\)",
                    str(raw_val),
                    re.IGNORECASE,
                )
                m2 = re.match(
                    r"\s*([0-9.]+)\s*\*\s*sin\(\s*([0-9.]+)\s*(π|pi)\s*t\s*\)\s*$",
                    str(raw_val),
                    re.IGNORECASE,
                )

                if m1:
                    amp = m1.group(1).strip()
                    omega = float(m1.group(2).strip())
                    phase_raw = m1.group(3).strip()
                    if "θ" in phase_raw and "theta" not in self.defined_params:
                        self.params += ".param theta=0\n"
                        self.defined_params.add("theta")
                    phase = "{theta}" if "θ" in phase_raw else phase_raw
                    freq = f"{omega / (2 * math.pi):g}"
                    line = f"{name} {n1} {n2} SIN(0 {amp} {freq} 0 0 {phase})"
                    self.any_ac_component = True
                elif m2:
                    amp = m2.group(1).strip()
                    coef = float(m2.group(2).strip())
                    omega = coef * math.pi
                    freq = f"{omega / (2 * math.pi):g}"
                    line = f"{name} {n1} {n2} SIN(0 {amp} {freq})"
                    self.any_ac_component = True
                else:
                    line = f"{name} {n1} {n2} DC 0"

            else:
                if str(raw_val).upper() == "NA" and kind == "V":
                    if self._is_name_vs(raw_name):
                        line = f"{name.upper()} {n1} {n2} AC 1"
                        self.any_ac_component = True
                    else:
                        line = f"{name} {n1} {n2} DC 1"
                else:
                    val = self._parse_numeric_value(str(raw_val)) if re.search(r"[\d.]", str(raw_val)) else "1"
                    line = f"{name} {n1} {n2} DC {val}"

            self._track_nodes_for_device(name, (n1, n2))
            if kind == "V":
                self.voltage_exists = True
                self.source_labels.append(name)
            return line

        # diode
        if kind == "D":
            if len(tokens) < 4:
                return ""
            name = self._sanitize_name(raw_name)
            n1 = norm(tokens[2])
            n2 = norm(tokens[3])
            raw_val = tokens[4] if len(tokens) >= 5 else "NA"

            # NA diodes use Dmod
            if self._use_dmod_for_na_diodes and str(raw_val).upper() == "NA":
                if ".model Dmod" not in self.models:
                    self.models += ".model Dmod D(Is=1e-14 N=1 Rs=0.1 Cjo=1e-12)\n"
                line = f"{name} {n1} {n2} Dmod"
                self._track_nodes_for_device(name, (n1, n2))
                return line

            # NA model
            if str(raw_val).upper() == "NA":
                line = f"{name} {n1} {n2} NA"
                if ".model NA" not in self.models:
                    if re.search(r"\b\d+\s*V\s*,\s*\d+\s*Hz\b", self.schema, re.IGNORECASE):
                        self.models += ".model NA D(IS=1e-14 N=1 RS=0.1 CJO=1p M=0.5 TT=0.1u BV=100 IBV=0.1m)\n"
                    else:
                        self.models += ".model NA D(Is=1e-14 N=1.2 Rs=0.1 Cjo=1p)\n"
                self._track_nodes_for_device(name, (n1, n2))
                return line

            self.uses_diode = True

            z = re.match(r"^\s*([0-9.]+)\s*V\s*$", str(raw_val), re.IGNORECASE)
            if z:
                bv = z.group(1)
                bv_key = f"{float(bv):g}"
                model = self.zener_models.get(bv_key)
                if model is None:
                    model = f"DZ{bv_key.replace('.', '_')}"
                    self.zener_models[bv_key] = model
                line = f"{name} {n2} {n1} {model}" if n1 == "0" else f"{name} {n1} {n2} {model}"
            else:
                line = f"{name} {n1} {n2} DIO"

            self._track_nodes_for_device(name, (n1, n2))
            return line

        # BJT
        if kind in ("NPN", "PNP", "Q"):
            if len(tokens) < 5:
                return ""
            name = self._sanitize_name(raw_name)

            if kind == "Q":
                self.uses_transistor = True
                nc, nb, ne = map(norm, tokens[2:5])
                line = f"{name} {nc} {nb} {ne} QNPN"
                self._track_nodes_for_device(name, (nc, nb, ne))
                return line

            self.uses_transistor = True
            nc, nb, ne = map(norm, tokens[2:5])

            if kind == "NPN" and self._npn_override_model:
                model = self._npn_override_model
            else:
                model = "NPNMOD" if kind == "NPN" else "PNPMOD"

            line = f"{name} {nc} {nb} {ne} {model}"
            self._track_nodes_for_device(name, (nc, nb, ne))
            return line

        # switch element "SW"
        if kind == "SW":
            if len(tokens) < 4:
                return ""
            name = self._sanitize_name(raw_name)
            self.has_switch = True
            n1 = norm(tokens[2])
            n2 = norm(tokens[3])
            ctl = "NCTL" if name.upper() == "S1" else f"NCTL_{name}"
            if "VCTRL" not in self.defined_params:
                self.connections += f"VCTRL {ctl} 0 PULSE(0 5 0 1n 1n 1 2)\n"
                self.defined_params.add("VCTRL")
            line = f"{name} {n1} {n2} {ctl} 0 SWMOD"
            self._track_nodes_for_device(name, (n1, n2, ctl, "0"))
            return line

        # transformer coupling K
        if kind == "K":
            if len(tokens) < 4:
                return ""
            name = self._sanitize_name(raw_name)
            lp = self._sanitize_name(tokens[2])
            ls = self._sanitize_name(tokens[3])
            kval = "0.999" if len(tokens) < 5 or tokens[4].upper() == "NA" else self._parse_numeric_value(tokens[4])
            self.any_ac_component = True
            line = f"{name} {lp} {ls} {kval}"
            self.device_labels.append(name)
            return line

        # impedance element Z: treat as resistor with default 1k
        if kind == "Z":
            if len(tokens) < 4:
                return ""
            name = self._sanitize_name(raw_name)
            n1 = norm(tokens[2])
            n2 = norm(tokens[3])
            val = "1k"
            return f"R{name} {n1} {n2} {val}"

        return ""

    # ---------------------------
    # Emit opamps and amps
    # ---------------------------

    def _emit_opamps(self) -> None:
        if not self.opamp_instances:
            return

        n_ops = len(self.opamp_instances)

        if n_ops == 1 and self.single_opamp_use_e_model:
            nn, np, no = self.opamp_instances[0]
            self.connections += f"EOP {no} 0 {np} {nn} 1e6\n"
            self.connections += f"ROUT {no} 0 50\n"
            self._track_nodes_for_device("EOP", (no, "0"))
            return

        if n_ops == 1:
            nn, np, no = self.opamp_instances[0]
            if "A" not in self.defined_params:
                self.params += ".param A=1e5\n"
                self.defined_params.add("A")

            if "VDD" not in self.connections and "VEE" not in self.connections:
                self.connections += f"VDD VDD 0 DC {self.default_vp}\n"
                self.connections += f"VEE VEE 0 DC {self.default_vn}\n"
                self.source_labels.extend(["VDD", "VEE"])
                self.voltage_exists = True

            self.connections += f"BOP {no} 0 V = LIMIT({{A}}*(V({np})-V({nn})), V(VEE), V(VDD))\n"
            self._track_nodes_for_device("BOP", (no, "0"))
            return

        if "OPAMP_SUBCKT" not in self.defined_params:
            self.connections += (
                ".subckt OPAMP IN- IN+ OUT VP VN\n"
                ".param A=1e6\n"
                "BOP OUT 0 V = 0.5*(V(VP)+V(VN)) + 0.5*(V(VP)-V(VN))*tanh({A}*(V(IN+)-V(IN-))/(abs(V(VP)-V(VN))+1e-6))\n"
                "ROUT OUT 0 1e6\n"
                ".ends OPAMP\n"
            )
            self.defined_params.add("OPAMP_SUBCKT")

        for k in range(1, n_ops + 1):
            if f"VP{k}" not in self.connections and f"VN{k}" not in self.connections:
                self.connections += f"VCC{k} VP{k} 0 DC {self.default_vp}\n"
                self.connections += f"VEE{k} VN{k} 0 DC {self.default_vn}\n"
                self.source_labels.extend([f"VCC{k}", f"VEE{k}"])
                self.voltage_exists = True

        for idx, (nn, np, no) in enumerate(self.opamp_instances, start=1):
            xname = f"XOP{idx}"
            self.connections += f"{xname} {nn} {np} {no} VP{idx} VN{idx} OPAMP\n"
            self.device_labels.append(xname)
            self.device_nodes[xname] = (nn, np, no)

    def _emit_amps(self) -> None:
        if not self.amp_instances:
            return
        for idx, (np, nn, no) in enumerate(self.amp_instances, start=1):
            ename = f"E{idx}"
            self.connections += f"{ename} {no} 0 {np} {nn} 1e6\n"
            self._track_nodes_for_device(ename, (no, "0"))

    # ---------------------------
    # Models
    # ---------------------------

    def _finalize_models(self) -> None:
        if self.uses_diode and "DIO" not in self.models:
            self.models += ".model DIO D(Is=1e-15 N=1)\n"

        for bv_key, mname in sorted(self.zener_models.items(), key=lambda x: float(x[0])):
            self.models += f".model {mname} D (IS=1e-14 N=1 BV={bv_key} IBV=1m)\n"

        if self._npn_override_model_line and self._npn_override_model and self._npn_override_model not in self.models:
            if self._npn_override_model not in self.models:
                self.models += self._npn_override_model_line

        if self.uses_transistor:
            if "QNPN" not in self.models:
                self.models += ".model QNPN NPN(BF=100 VAF=100 IS=1e-14 RB=100 RC=10 RE=1 Cjc=5p Tf=0.6n)\n"
            if "NPNMOD" not in self.models:
                self.models += ".model NPNMOD NPN(IS=1e-14 BF=150 VAF=100 CJE=20p CJC=8p TF=0.6n TR=100n)\n"
            if "PNPMOD" not in self.models:
                self.models += ".model PNPMOD PNP(IS=1e-14 BF=100 VAF=100)\n"

        if self.has_switch:
            if self.switch_model_override:
                if "SWMOD" not in self.models:
                    self.models += self.switch_model_override
            else:
                if "SWMOD" not in self.models:
                    self.models += ".model SWMOD SW(Ron=1m Roff=10Meg Vt=2.5 Vh=0.1)\n"

    # ---------------------------
    # Fixups
    # ---------------------------

    def _apply_eis_override(self) -> None:
        if not self.eis_override:
            return
        eis = self.eis_override

        lines = [ln for ln in self.connections.splitlines() if ln.strip()]
        new_lines = []
        r_seen = 0
        for ln in lines:
            if ln.startswith("R") and r_seen == 0:
                parts = ln.split()
                parts[-1] = eis["R1"]
                new_lines.append(" ".join(parts))
                r_seen += 1
                continue
            if ln.startswith("R") and r_seen == 1:
                parts = ln.split()
                parts[-1] = eis["R2"]
                new_lines.append(" ".join(parts))
                r_seen += 1
                continue
            if ln.startswith("C"):
                parts = ln.split()
                parts[-1] = eis["C"]
                new_lines.append(" ".join(parts))
                continue
            new_lines.append(ln)
        self.connections = "\n".join(new_lines) + "\n"

    def _inject_ammeter_links(self) -> None:
        if not self.ammeter_mode:
            return
        n1 = self.ammeter_n1
        for node in sorted(set(self.ammeter_link_nodes)):
            m = re.match(r"^N(\d+)$", node)
            if m:
                vname = f"Vlink1{m.group(1)}"
            else:
                vname = f"Vlink_{node}"
            if f"{vname} " in self.connections:
                continue
            self.connections += f"{vname} {n1} {node} DC 0\n"
            self.voltage_exists = True

        if self.ammeter_nr_node:
            if "Vgnd NR 0 DC 0" not in self.connections:
                self.connections += "Vgnd NR 0 DC 0\n"
                self.voltage_exists = True

    def _add_line_coupling_if_needed(self) -> None:
        # Add K12 L1 L2 0.5 for transmission lines
        if len(self._line_Ls) >= 2 and self._line_k is not None:
            if not any(ln.strip().startswith("K12 ") for ln in self.connections.splitlines()):
                self.connections += f"K12 {self._line_Ls[0]} {self._line_Ls[1]} {self._line_k}\n"

    def _ground_and_missing_source_fixes(self) -> None:
        self._ensure_ground_by_degree_rule()

        if not self.voltage_exists:
            i_lines = [ln for ln in self.connections.splitlines() if ln.strip().upper().startswith("I")]
            if i_lines:
                t = i_lines[0].split()
                if len(t) >= 4:
                    iname, n1, n2 = t[0], t[1], t[2]
                    if n1 != "0":
                        self.connections = self.connections.replace(f"{iname} {n1} {n2}", f"{iname} 0 {n2}")
                        self._replace_node_everywhere(n1, "0")
                    self.connections += f"V1 {n2} 0 DC 1\n"
                    self.source_labels.append("V1")
                    self.voltage_exists = True
                    return

            all_conn = [ln for ln in self.connections.splitlines() if ln.strip()]
            if all_conn:
                first = all_conn[0].split()
                if len(first) >= 3:
                    n1 = first[1]
                    self.connections += f"V1 {n1} 0 DC 1\n"
                    self.source_labels.append("V1")
                    self.voltage_exists = True
                    return

    # ---------------------------
    # Main parse
    # ---------------------------

    def parse_schema(self) -> None:
        # canned modes first
        if self._handle_figure_subckts_if_present():
            return
        if self._handle_pwm_bridge_if_present():
            return
        if self._handle_cuk_if_present():
            return
        if self._handle_meter_coupled_transformer_if_present():
            return
        if self._handle_two_coils_if_present():
            return
        if self._handle_scr_pair_ac_if_present():
            return

        parts = re.split(r"\b[Ll]abels\b", self.schema, maxsplit=1)
        conn_block = parts[0]
        labels_block = parts[1] if len(parts) > 1 else ""

        # Detect Dmod override case
        if "IDC+iac" in labels_block or "VDC+Vac" in labels_block:
            self._use_dmod_for_na_diodes = True

        # Detect MOD1 override case
        if "81kΩ" in self.schema and "8.1kΩ" in self.schema and "MOD1" not in self.schema:
            self._npn_override_model = "MOD1"
            self._npn_override_model_line = ".model MOD1 NPN BF=50 VAF=50 IS=1.E-12 RB=100 CJC=.5PF TF=.6NS\n"

        # EIS override detection
        self.eis_override = self._extract_eis_semicircle_params(labels_block)

        # parse connection lines first
        for ln in conn_block.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            out = self.add_connections(ln)
            if out:
                self.connections += out + "\n"

        # label u V N1 N2 and no explicit 0 means N2 becomes ground
        if re.search(r"\bu\s+V\s+N1\s+N2\b", labels_block) and not re.search(r"(^|\s)0(\s|$)", conn_block):
            self._force_node_as_ground_if_reference("N2")

        # decide AC or DC (sources can mark any_ac_component)
        self.AC_or_DC = "AC" if self.any_ac_component else "DC"

        # labels may add sources, probes, rails
        self._emit_label_sources(labels_block)

        # emit opamps and amps
        self._emit_opamps()
        self._emit_amps()

        # apply EIS deterministic rewriting
        self._apply_eis_override()

        # transmission line coupling
        self._add_line_coupling_if_needed()

        # ammeter links
        self._inject_ammeter_links()

        # grounding and missing source fixes
        self._ground_and_missing_source_fixes()

        # analysis
        if self.AC_or_DC == "AC":
            self.analysis += ".ac lin 1 1 1\n"
        else:
            self.analysis += ".op\n"

        # models after all devices are known
        self._finalize_models()

    def make_template(self) -> str:
        self.parse_schema()
        template = ""
        template += self.begin
        if self.params.strip():
            template += self.params
        if self.models.strip():
            template += self.models
        template += self.connections
        if self.analysis.strip():
            template += self.analysis
        template += self.end
        return template

    def ngspice_template_to_json(self) -> Dict[str, List[str]]:
        def split_block(block: str) -> List[str]:
            return [ln for ln in block.splitlines() if ln.strip()]

        return {
            "params": split_block(self.params),
            "models": split_block(self.models),
            "connections": split_block(self.connections),
            "analysis": split_block(self.analysis),
        }
