"""Prompt template for the joint component/connectivity reasoning step.

The VLM is shown three images of the same schematic (plain text-erased, marked
text-erased, and text-boxed original -- see extract.py) plus deterministic
wire-tracing evidence, and asked to produce one structured netlist in a single
call.
"""

# Closed type vocabulary, derived from scanning a large sample of gold circuit
# schemas. Giving the model this exact vocabulary means its output can be
# compared to gold types without fuzzy string matching.
TYPE_VOCAB = [
    "R", "V", "C", "D", "L", "I", "Q", "NPN", "PNP", "S", "SW", "M", "NMOS",
    "PMOS", "AMP", "OPAMP", "A", "Z", "SCR", "AMMETER", "VOLTMETER", "VM",
    "AM", "LAMP", "OTHER",
]

_OUTPUT_SPEC = f"""
Output STRICT JSON only, no prose, no markdown fences, matching this schema:
{{
  "components": [{{"id": "<string>", "type": "<one of: {', '.join(TYPE_VOCAB)}>"}}],
  "connections": [["<id_a>", "<id_b>"], ...]
}}

Rules:
- "id" must be a short unique label you assign to each component you see (e.g. "R1", "C1").
- "type" must be exactly one token from the allowed type list above. Use "OTHER" if unsure.
- "connections" lists pairs of component ids that are directly joined by a wire
  (if two components are joined only through a chain of wires with no other component
  in between, still count them as connected).
- Do not include ground/net names as components; only physical circuit elements.
- Do not output anything except the JSON object.
"""


def build_extraction_prompt(wire_summary: str) -> tuple[str, str]:
    """The single joint-reasoning call: given the three component/text-evidence
    images (attached separately, see extract.py) and deterministic wire-tracing
    evidence, produce the full netlist in one shot."""
    system_prompt = (
        "You are an expert at reading electrical circuit schematic diagrams and "
        "extracting their structure as a component/connection graph."
    )
    image_note = (
        "You are given three images of the same schematic: (1) the plain diagram "
        "with all text erased, for reading the circuit's shape cleanly; (2) the "
        "same erased diagram with numbered markers circled at candidate component "
        "locations, for resolving marker ids in the evidence below; (3) the "
        "original, unerased diagram with a red outline around every detected text "
        "region (the text itself is fully legible), for reading component labels "
        "and values where needed."
    )
    draft_block = (
        "Using the images and the evidence below, produce the component/connection "
        "list for this circuit."
    )
    marker_mapping_note = (
        "The wire-tracing evidence below refers to components by their numbered "
        "marker id (as circled in image 2), not by the component id you assign "
        "them. Before you can use that evidence, you must first output a field "
        "\"marker_mapping\": an object mapping every numbered marker id shown in "
        "image (2) to a component id already present in your \"components\" list "
        "(e.g. {\"1\": \"R1\", \"2\": \"C1\", ...}). Every marker circled in image "
        "(2) must appear as a key. IMPORTANT: you must map each marker to an "
        "existing component id from your list, or to \"NONE\" -- never invent a "
        "new component id here that is not already in \"components\". A marker "
        "that doesn't clearly belong to any component you listed (e.g. it lands "
        "on a wire, a junction, or empty space) must be mapped to \"NONE\", not to "
        "a new made-up component -- do not add new components just to have "
        "something to map a marker to."
    )
    evidence_check = (
        "Using the marker_mapping you just wrote, translate every high-confidence "
        "wire-tracing connection below (given as a pair of marker ids) into a pair "
        "of your own component ids, skipping any pair where either marker maps to "
        "\"NONE\" (never add \"NONE\" to your connections list). Go through this "
        "translated list one connection at a time and check it against your "
        "\"connections\" list: if a translated high-confidence connection is "
        "missing, you must add it to \"connections\" -- this is a required "
        "correction, not optional, unless the images clearly show the two "
        "components are not actually joined by a wire. Do not simply relabel or "
        "reformat without performing this translate-and-check step -- you must "
        "actually use the evidence, not just describe what you see in the images."
    )
    bridge_rectifier_note = (
        "If you see four diodes arranged in a diamond/box pattern between an AC "
        "source (or transformer secondary) and a DC load (a bridge rectifier), list "
        "all four diodes as separate components (e.g. D1, D2, D3, D4), not as one "
        "combined \"bridge\" or \"rectifier\" component."
    )
    schema_addendum = (
        "\nYour JSON output for this step must include a third top-level field in "
        "addition to \"components\" and \"connections\":\n"
        "  \"marker_mapping\": {\"<marker_id>\": \"<component_id or NONE>\", ...}\n"
        "as described above -- this is required for this step, on top of the "
        "schema shown above."
    )
    user_prompt = (
        f"{image_note}\n\n{draft_block}\n\n"
        "Automated wire-tracing analysis (may be incomplete or wrong in places):\n"
        f"{wire_summary}\n\n"
        "Treat the wire-tracing analysis as evidence, not ground truth.\n\n"
        f"{marker_mapping_note}\n\n{evidence_check}\n\n{bridge_rectifier_note}"
        + _OUTPUT_SPEC + schema_addendum
    )
    return system_prompt, user_prompt
