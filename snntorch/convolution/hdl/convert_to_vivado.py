#!/usr/bin/env python3
"""
Improved script to convert VUnit testbench files to Vivado-compatible simulation files.
Strips VUnit framework and converts to standalone testbench with better syntax handling.
"""

import re
import argparse
from pathlib import Path

def extract_test_cases(content):
    """Extract all test case names from if run() statements."""
    test_cases = []
    pattern = r'if\s+run\s*\(\s*"([^"]+)"\s*\)'
    matches = re.finditer(pattern, content, re.IGNORECASE)
    
    for match in matches:
        test_name = match.group(1)
        if test_name not in test_cases:
            test_cases.append(test_name)
    
    return test_cases

def extract_test_code(content, test_name):
    """Extract the actual test code for a specific test case with better syntax handling."""
    # Find the start of the test case
    start_pattern = rf'if\s+run\s*\(\s*"{re.escape(test_name)}"\s*\)\s+then\s*'
    start_match = re.search(start_pattern, content, re.IGNORECASE)
    
    if not start_match:
        # Try elsif pattern
        start_pattern = rf'elsif\s+run\s*\(\s*"{re.escape(test_name)}"\s*\)\s+then\s*'
        start_match = re.search(start_pattern, content, re.IGNORECASE)
    
    if not start_match:
        return f"-- Test code for {test_name} not found"
    
    # Find the content between the test start and the next test or end if
    start_pos = start_match.end()
    content_after = content[start_pos:]
    
    # Look for the end of this test case
    end_patterns = [
        r'\s*elsif\s+run\s*\(',  # Next test case
        r'\s*end\s+if\s*;',      # End of all tests
        r'\s*test_runner_cleanup'  # VUnit cleanup
    ]
    
    end_pos = len(content_after)
    for pattern in end_patterns:
        match = re.search(pattern, content_after, re.IGNORECASE)
        if match:
            end_pos = min(end_pos, match.start())
    
    test_code = content_after[:end_pos].strip()
    
    # Clean up the extracted code
    test_code = clean_test_code(test_code)
    
    return test_code

def clean_test_code(code):
    """Clean up extracted test code by removing problematic constructs."""
    lines = code.split('\n')
    cleaned_lines = []
    
    skip_keywords = [
        'function ', 'procedure ', 'component ', 'entity ', 'architecture ',
        'signal ', 'constant ', 'type ', 'subtype ', 'package ',
        'library ', 'use ', 'end function', 'end procedure', 
        'end entity', 'end architecture', 'end component'
    ]
    
    for line in lines:
        line_stripped = line.strip().lower()
        
        # Skip VHDL declarations that shouldn't be in a process
        should_skip = any(line_stripped.startswith(keyword) for keyword in skip_keywords)
        should_skip = should_skip or 'generic map' in line_stripped or 'port map' in line_stripped
        
        if not should_skip:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def extract_procedures_and_functions(content):
    """Extract procedure and function definitions from the testbench."""
    procedures = []
    functions = []
    
    # Extract procedures
    proc_pattern = r'procedure\s+\w+.*?end\s+procedure\s*;'
    proc_matches = re.finditer(proc_pattern, content, re.DOTALL | re.IGNORECASE)
    for match in proc_matches:
        procedures.append(match.group(0))
    
    # Extract functions  
    func_pattern = r'function\s+\w+.*?end\s+function\s*;'
    func_matches = re.finditer(func_pattern, content, re.DOTALL | re.IGNORECASE)
    for match in func_matches:
        functions.append(match.group(0))
    
    return procedures, functions

def convert_testbench_to_vivado(input_file, output_file):
    """Convert VUnit testbench to Vivado-compatible standalone testbench."""
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    print(f"Converting: {input_file}")
    
    # Extract test cases before removing VUnit code
    test_cases = extract_test_cases(content)
    print(f"Found test cases: {test_cases}")
    
    # Extract procedures and functions before modification
    procedures, functions = extract_procedures_and_functions(content)
    
    # Remove VUnit imports
    content = re.sub(r'library\s+vunit_lib\s*;\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'context\s+vunit_lib\.vunit_context\s*;\s*', '', content, flags=re.IGNORECASE)
    
    # Extract entity name
    entity_match = re.search(r'entity\s+(\w+)\s+is', content, re.IGNORECASE)
    entity_name = entity_match.group(1) if entity_match else "testbench"
    print(f"Entity name: {entity_name}")
    
    # Remove runner_cfg generic from entity
    entity_pattern = rf'entity\s+{re.escape(entity_name)}\s+is\s*generic\s*\(\s*runner_cfg\s*:\s*string\s*\)\s*;\s*end\s+entity\s+{re.escape(entity_name)}\s*;'
    content = re.sub(
        entity_pattern,
        f'entity {entity_name} is\nend entity {entity_name};',
        content,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    
    # Fix clock generation
    clock_replacement = '''clk_process: process
    begin
        clk <= '1';
        wait for CLK_PERIOD/2;
        clk <= '0';
        wait for CLK_PERIOD/2;
    end process;'''
    
    content = re.sub(
        r'clk\s*<=\s*not\s+clk\s+after\s+\d+\s*ns\s*;',
        clock_replacement,
        content,
        flags=re.MULTILINE
    )
    
    # Extract main process
    main_process_pattern = r'main\s*:\s*process\s*(?:\([^)]*\))?\s*(.*?)\s*begin(.*?)end\s+process\s+main\s*;'
    main_match = re.search(main_process_pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    
    variable_declarations = ""
    if main_match:
        print("Found main process")
        potential_vars = main_match.group(1).strip()
        if potential_vars and ('variable' in potential_vars.lower() or 'constant' in potential_vars.lower()):
            variable_declarations = f"        {potential_vars}\n"
            print(f"Found variable declarations")
    else:
        print("Warning: Main process not found")
    
    # Build new main process
    new_main_process = "main : process\n"
    if variable_declarations:
        new_main_process += variable_declarations
    new_main_process += "    begin\n\n"
    new_main_process += "        -- Initial stabilization\n"
    new_main_process += "        wait_cycles(10);\n\n"
    
    # Add each test
    for test_name in test_cases:
        test_code = extract_test_code(content, test_name)
        if "not found" not in test_code:
            print(f"Extracted test: {test_name}")
        else:
            print(f"Warning: {test_code}")
        
        new_main_process += f"        -- Test: {test_name}\n"
        new_main_process += f"        report \"Running test: {test_name}\";\n"
        
        # Indent the test code properly
        indented_test_code = '\n'.join(['        ' + line if line.strip() else line 
                                       for line in test_code.split('\n')])
        new_main_process += f"{indented_test_code}\n"
        new_main_process += f"        report \"Test {test_name} completed\";\n\n"
    
    new_main_process += "        report \"All tests completed successfully\";\n"
    new_main_process += "        wait;\n\n"
    new_main_process += "    end process main;"
    
    # Replace main process
    if main_match:
        content = re.sub(
            main_process_pattern,
            new_main_process,
            content,
            flags=re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        print("Replaced main process successfully")
    else:
        print("Warning: Could not find main process to replace")
        # Insert before end architecture
        content = re.sub(
            r'(end\s+(?:testbench|architecture)\s*;)',
            new_main_process + '\n\n\\1',
            content,
            flags=re.IGNORECASE
        )
        print("Inserted new main process")
    
    # Clean up VUnit artifacts
    content = re.sub(r'\s*test_runner_setup\([^)]*\)\s*;\s*', '', content)
    content = re.sub(r'\s*test_runner_cleanup\([^)]*\)\s*;\s*', '', content)
    
    # Remove stray VUnit constructs but be more careful
    content = re.sub(r'\s*if\s+run\s*\([^)]*\)\s+then\s*', '', content)
    content = re.sub(r'\s*elsif\s+run\s*\([^)]*\)\s+then\s*', '', content)
    
    # Clean up excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    content = re.sub(r'^\s*$\n', '', content, flags=re.MULTILINE)
    
    # Ensure proper VHDL structure
    content = ensure_proper_syntax(content)
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Converted testbench saved to: {output_file}")
    return test_cases

def ensure_proper_syntax(content):
    """Ensure proper VHDL syntax by checking for common issues."""
    lines = content.split('\n')
    result_lines = []
    
    for line in lines:
        # Skip empty lines at the beginning
        if not result_lines and not line.strip():
            continue
            
        result_lines.append(line)
    
    return '\n'.join(result_lines)

def main():
    """Main conversion function."""
    
    parser = argparse.ArgumentParser(description="Convert VUnit testbench to Vivado-compatible simulation file")
    parser.add_argument("input_file", help="Input VUnit testbench file (.vhd)")
    parser.add_argument("-o", "--output", help="Output file (default: vivado_<input_file>)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        return 1
    
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"vivado_{input_file.name}"
    
    print("VUnit to Vivado Testbench Converter (Improved)")
    print("=" * 50)
    
    try:
        test_cases = convert_testbench_to_vivado(str(input_file), str(output_file))
        
        print("=" * 50)
        print("Conversion Summary:")
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print(f"Tests converted: {len(test_cases)}")
        for test in test_cases:
            print(f"  - {test}")
        print("\nConversion completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())