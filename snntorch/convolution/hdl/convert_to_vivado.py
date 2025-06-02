#!/usr/bin/env python3
"""
convert_to_vivado.py

A script to convert VUnit-based VHDL testbenches into Vivado-compatible standalone testbenches.
It strips out VUnit-specific constructs, extracts test cases, fixes a known pooling-typo,
and rebuilds a main process that sequentially runs each test, emitting REPORT statements
for waveform inspection. It also inserts a default clock generator if none exists.
"""

import re
import argparse
from pathlib import Path

def extract_test_cases(content):
    """
    Extract all test case names from `if run("test_name")` or `elsif run("test_name")` statements.
    Returns a list of unique test names in the order found.
    """
    pattern = r'\bif\s+run\s*\(\s*"([^"]+)"\s*\)\s+then'
    names = []
    for m in re.finditer(pattern, content, flags=re.IGNORECASE):
        name = m.group(1)
        if name not in names:
            names.append(name)
    return names

def extract_test_code(content, test_name):
    """
    Extract the VHDL statements inside the `if run("test_name") then ...` block.
    Strips leading/trailing whitespace and removes any nested `if run` or `elsif run` markers.
    Returns the code as a string (without the wrapping 'if/elsif').
    """
    # Locate the start of this test case
    start_pattern = rf'\b(if|elsif)\s+run\s*\(\s*"{re.escape(test_name)}"\s*\)\s+then'
    start_m = re.search(start_pattern, content, flags=re.IGNORECASE)
    if not start_m:
        return f"-- [WARNING] Test code for \"{test_name}\" not found"
    
    start_pos = start_m.end()
    # Now find where this block ends: next `elsif run(`, `end if;`, or `test_runner_cleanup`
    remainder = content[start_pos:]
    end_patterns = [
        r'\belsif\s+run\s*\(',     # next test
        r'\bend\s+if\s*;',         # end of tests region
        r'\btest_runner_cleanup\b' # end marker
    ]
    end_pos = len(remainder)
    for pat in end_patterns:
        m = re.search(pat, remainder, flags=re.IGNORECASE)
        if m and m.start() < end_pos:
            end_pos = m.start()
    block = remainder[:end_pos].strip()
    return clean_test_code(block)

def clean_test_code(code):
    """
    Remove declarations or VUnit-specific calls inside the extracted test code.
    Skips lines that begin with typical declaration keywords or VUnit runner calls.
    """
    skip_keywords = [
        'function ', 'procedure ', 'component ', 'entity ',
        'architecture ', 'signal ', 'constant ', 'type ', 'subtype ',
        'package ', 'library ', 'use ', 'end function', 'end procedure',
        'end entity', 'end architecture', 'end component'
    ]
    lines = code.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip().lower()
        # Skip if line starts with any declaration keyword
        if any(stripped.startswith(kw) for kw in skip_keywords):
            continue
        # Skip runner calls
        if 'test_runner_setup' in stripped or 'test_runner_cleanup' in stripped:
            continue
        # Keep everything else
        cleaned.append(line.rstrip())
    return '\n'.join(cleaned).rstrip()

def extract_declarative_region(content):
    """
    Split the architecture into:
      1) The line 'architecture <name> of <entity> is'
      2) Everything up to the first 'begin' in that architecture (declarations region)
      3) The 'begin' keyword
      4) Everything after that 'begin'
    Returns (arch_head, declarations_text, begin_keyword, rest_after_begin).
    If not found, returns (None, None, None, None).
    """
    pattern = r'(architecture\s+\w+\s+of\s+\w+\s+is)(.*?)(\bbegin\b)(.*)'
    m = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None, None, None, None
    return m.group(1), m.group(2), m.group(3), m.group(4)

def remove_runner_cfg_generic(entity_block):
    """
    Given the entity declaration block (up to 'end entity;'),
    remove only the runner_cfg generic if present, preserving other generics.
    """
    gen_pattern = r'(generic\s*\(.*?\))'
    m = re.search(gen_pattern, entity_block, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return entity_block
    full_generic = m.group(1)
    # Remove runner_cfg : string [;]
    new_generic = re.sub(r'runner_cfg\s*:\s*string\s*;?', '', full_generic, flags=re.IGNORECASE)
    # If nothing left inside parentheses, drop the entire generic(...) block
    inside = new_generic[new_generic.find('(')+1:new_generic.rfind(')')].strip()
    if not inside:
        return entity_block.replace(full_generic, '')
    else:
        return entity_block.replace(full_generic, new_generic)

def remove_vunit_stuff(content):
    """
    Strip out common VUnit-specific constructs:
      - library vunit_lib;
      - context vunit_lib.vunit_context;
      - test_runner_setup(...);
      - test_runner_cleanup(...);
      - any 'if run(...) then' / 'elsif run(...) then'
    Returns the cleaned content.
    """
    patterns = [
        r'^\s*library\s+vunit_lib\s*;.*?$',                             # library vunit_lib;
        r'^\s*context\s+vunit_lib\.vunit_context\s*;.*?$',              # context vunit_lib.vunit_context;
        r'\btest_runner_setup\s*\([^\)]*\)\s*;?',                       # test_runner_setup(...)
        r'\btest_runner_cleanup\s*\([^\)]*\)\s*;?',                     # test_runner_cleanup(...)
        r'\bif\s+run\s*\([^\)]*\)\s+then',                              # if run("...") then
        r'\belsif\s+run\s*\([^\)]*\)\s+then',                           # elsif run("...") then
        # We deliberately leave 'end if;' to help bracket test blocks, then remove later if needed
    ]
    for pat in patterns:
        content = re.sub(pat, '', content, flags=re.IGNORECASE | re.MULTILINE)
    return content

def replace_main_process(content, new_main_body):
    """
    Replace the existing main process definition (main: process ... begin ... end process main;)
    with new_main_body. If not found, insert new_main_body right before 'end architecture ...;'.
    """
    pattern = r'(main\s*:\s*process\b.*?\bbegin\b)(.*?)(end\s+process\s+main\s*;)'
    m = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
    if m:
        prefix = m.group(1)
        suffix = m.group(3)
        return re.sub(pattern,
                      f"{prefix}\n{new_main_body}\n{suffix}",
                      content, flags=re.IGNORECASE | re.DOTALL)
    else:
        # If main process not found, insert before end of architecture
        insert_pattern = r'(end\s+architecture\s+\w+\s*;)'
        return re.sub(insert_pattern, f"{new_main_body}\n\n\\1", content, flags=re.IGNORECASE)

def ensure_proper_syntax(content):
    """
    Simple cleanup: remove leading blank lines, collapse multiple blank lines.
    """
    # Remove leading blank lines
    content = re.sub(r'^\s*\n', '', content)
    # Collapse 3+ blank lines into 2
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content

def convert_testbench_to_vivado(input_path, output_path):
    """
    Main function to read the VUnit testbench, strip VUnit parts, extract tests,
    fix a pooling typo, insert a clock generator if missing, and reassemble a Vivado-friendly testbench.
    """
    raw = Path(input_path).read_text()

    # ------------------------------------------------------------
    # 0) Fix known pooling-typo: replace any occurrence of "...IGHT/POOL_SIZE"
    #    with "(IMG_HEIGHT/POOL_SIZE)". The user had a line like:
    #      (IMG_WIDTH/POOL_SIZE) * (...IGHT/POOL_SIZE)
    #    We substitute it so Vivado won't error.
    raw = re.sub(
        r'\(\s*IMG_WIDTH\s*/\s*POOL_SIZE\s*\)\s*\*\s*\(\s*\.\.\.IGHT\s*/\s*POOL_SIZE\s*\)',
        '(IMG_WIDTH/POOL_SIZE) * (IMG_HEIGHT/POOL_SIZE)',
        raw
    )
    # ------------------------------------------------------------

    # 1) Extract entity block and strip runner_cfg generic only
    ent_pattern = r'(entity\s+\w+\s+is.*?end\s+entity\s+\w+\s*;)'
    ent_m = re.search(ent_pattern, raw, flags=re.IGNORECASE | re.DOTALL)
    if ent_m:
        entity_block = ent_m.group(1)
        new_entity_block = remove_runner_cfg_generic(entity_block)
        raw = raw.replace(entity_block, new_entity_block)
    else:
        print(f"[Warning] No entity block found in {input_path}")

    # 2) Extract test cases before stripping any VUnit code
    test_cases = extract_test_cases(raw)
    print(f"Found test cases: {test_cases}")

    # 3) Extract and preserve any declarations from the architecture head
    arch_head, decl_region, begin_kw, after_begin = extract_declarative_region(raw)
    preserved_decls = ""
    if decl_region:
        # Remove any VUnit artifacts from the decl region, but keep user procedures/functions/constants
        preserved_decls = remove_vunit_stuff(decl_region)
    else:
        print("[Warning] Could not extract architecture declarative region.")

    # 4) Remove VUnit artifacts globally
    body_no_vunit = remove_vunit_stuff(raw)

    # 5) Build the new main process body
    lines = []
    lines.append("    -- Variables (if any) were declared above")
    lines.append("    begin")
    lines.append("")
    lines.append("        -- Initial stabilization (wait some time or cycles)")
    lines.append("        wait for 100 ns;  -- Adjust as needed")
    lines.append("")
    for test_name in test_cases:
        code_block = extract_test_code(body_no_vunit, test_name)
        lines.append(f"        -- Running test: {test_name}")
        lines.append(f"        report \"[INFO] Starting test: {test_name}\";")
        # Indent each line of the code block by 8 spaces
        for src_line in code_block.splitlines():
            if src_line.strip():
                lines.append("        " + src_line.rstrip())
        lines.append(f"        report \"[INFO] Test {test_name} completed\";")
        lines.append("")
    lines.append("        report \"[INFO] All tests completed successfully\";")
    lines.append("        wait;")
    lines.append("    end process main;")
    new_main_body = "\n".join(lines)

    # 6) Replace or insert the main process
    new_content = replace_main_process(body_no_vunit, new_main_body)

    # 7) Reconstruct architecture: reassemble arch_head + preserved_decls + begin + rest_after_begin
    if arch_head and begin_kw:
        arch_pattern_full = r'architecture\s+\w+\s+of\s+\w+\s+is.*?end\s+architecture\s+\w+\s*;'
        rebuilt_arch = f"{arch_head}\n{preserved_decls}\n{begin_kw}\n{after_begin}"
        new_content = re.sub(arch_pattern_full, rebuilt_arch, new_content, flags=re.IGNORECASE | re.DOTALL)

    # ------------------------------------------------------------
    # 8) Insert default clock process if none exists (i.e., no 'clk <= '1'' found)
    if not re.search(r"clk\s*<=\s*'1'", new_content):
        clock_block = """
    ----------------------------------------------------------------------------
    -- Clock Generator inserted by convert_to_vivado
    ----------------------------------------------------------------------------
    clk_process: process
    begin
        clk <= '1';
        wait for CLK_PERIOD/2;
        clk <= '0';
        wait for CLK_PERIOD/2;
    end process clk_process;

"""
        # Insert clock_block immediately after the first 'begin' in the architecture
        new_content = re.sub(
            r'(architecture\s+\w+\s+of\s+\w+\s+is[\s\S]*?\bbegin\b)',
            r"\1\n" + clock_block,
            new_content,
            flags=re.IGNORECASE
        )
    # ------------------------------------------------------------

    # 9) Final whitespace and syntax cleanup
    final_text = ensure_proper_syntax(new_content)

    # 10) Write to output file
    Path(output_path).write_text(final_text)
    print(f"Converted testbench saved to: {output_path}")
    return test_cases

def main():
    parser = argparse.ArgumentParser(
        description="Convert a VUnit VHDL testbench into a Vivado-compatible standalone testbench."
    )
    parser.add_argument("input_file", help="Input VUnit testbench file (.vhd or .vhdl)")
    parser.add_argument(
        "-o", "--output", metavar="OUTPUT_FILE",
        help="Output Vivado-friendly testbench (default: vivado_<input_file name>)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"vivado_{input_path.name}"

    print("VUnit → Vivado Testbench Converter")
    print("=" * 50)
    try:
        test_cases = convert_testbench_to_vivado(input_path, output_path)
        print("=" * 50)
        print(f"Input : {input_path}")
        print(f"Output: {output_path}")
        print(f"Tests converted ({len(test_cases)}):")
        for t in test_cases:
            print(f"  - {t}")
        print("\nConversion completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
