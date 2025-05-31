#!/usr/bin/env python3
"""
Script to convert VUnit testbench files to Vivado-compatible simulation files.
Strips VUnit framework and converts to standalone testbench.
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
    """Extract the actual test code for a specific test case."""
    # Pattern to match if run("test_name") then ... end if or elsif
    pattern = rf'if\s+run\s*\(\s*"{re.escape(test_name)}"\s*\)\s+then(.*?)(?=elsif|end\s+if)'
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Try elsif pattern
    pattern = rf'elsif\s+run\s*\(\s*"{re.escape(test_name)}"\s*\)\s+then(.*?)(?=elsif|end\s+if)'
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    return f"-- Test code for {test_name} not found"

def convert_testbench_to_vivado(input_file, output_file):
    """Convert VUnit testbench to Vivado-compatible standalone testbench."""
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    print(f"Converting: {input_file}")
    
    # Extract test cases before removing VUnit code
    test_cases = extract_test_cases(content)
    print(f"Found test cases: {test_cases}")
    
    # Remove VUnit library and context imports
    content = re.sub(r'library\s+vunit_lib\s*;\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'context\s+vunit_lib\.vunit_context\s*;\s*', '', content, flags=re.IGNORECASE)
    
    # Extract entity name
    entity_match = re.search(r'entity\s+(\w+)\s+is', content, re.IGNORECASE)
    entity_name = entity_match.group(1) if entity_match else "testbench"
    
    # Remove runner_cfg generic from entity
    content = re.sub(
        rf'entity\s+{re.escape(entity_name)}\s+is\s*generic\s*\(\s*runner_cfg\s*:\s*string\s*\)\s*;\s*end\s+entity\s+{re.escape(entity_name)}\s*;',
        f'entity {entity_name} is\nend entity {entity_name};',
        content,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    
    # Fix clock generation - replace concurrent assignment with process
    content = re.sub(
        r'clk\s*<=\s*not\s+clk\s+after\s+\d+\s*ns\s*;',
        '''clk_process: process
    begin
        clk <= '1';
        wait for CLK_PERIOD/2;
        clk <= '0';
        wait for CLK_PERIOD/2;
    end process;''',
        content,
        flags=re.MULTILINE
    )
    
    # Remove debug vector signals and port mappings (Vivado compatibility)
    content = re.sub(
        r'\s*signal\s+[^;]*debug\w*vec[^;]*:\s*std_logic_vector[^;]*;',
        '',
        content,
        flags=re.MULTILINE
    )
    
    content = re.sub(
        r',\s*debug\w*vec\s*=>\s*\w+',
        '',
        content,
        flags=re.MULTILINE
    )
    
    # Build new main process
    new_main_process = "main : process\n    begin\n\n"
    new_main_process += "        -- Initial stabilization\n"
    new_main_process += "        waitf(10);\n\n"
    
    # Add each test sequentially
    for test_name in test_cases:
        test_code = extract_test_code(content, test_name)
        new_main_process += f"        -- Test: {test_name}\n"
        new_main_process += f"        report \"Running test: {test_name}\";\n"
        new_main_process += f"        {test_code}\n"
        new_main_process += f"        report \"Test {test_name} completed\";\n\n"
    
    new_main_process += "        report \"All tests completed successfully\";\n"
    new_main_process += "        wait;\n\n"
    new_main_process += "    end process main;"
    
    # Replace the entire main process
    main_process_pattern = r'main\s*:\s*process\s*begin.*?end\s+process\s+main\s*;'
    content = re.sub(
        main_process_pattern,
        new_main_process,
        content,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    
    # Clean up any remaining VUnit artifacts
    content = re.sub(r'\s*test_runner_setup\([^)]*\)\s*;\s*', '', content)
    content = re.sub(r'\s*test_runner_cleanup\([^)]*\)\s*;\s*', '', content)
    content = re.sub(r'\s*while\s+test_suite\s+loop.*?end\s+loop\s*;\s*', '', content, flags=re.DOTALL)
    
    # Remove any remaining run() function calls
    content = re.sub(r'run\s*\(\s*"[^"]+"\s*\)', 'true', content)
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Converted testbench saved to: {output_file}")
    return test_cases

def main():
    """Main conversion function."""
    
    parser = argparse.ArgumentParser(description="Convert VUnit testbench to Vivado-compatible simulation file")
    parser.add_argument("input_file", help="Input VUnit testbench file (.vhd)")
    parser.add_argument("-o", "--output", help="Output file (default: vivado_<input_file>)")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        return
    
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"vivado_{input_file.name}"
    
    print("VUnit to Vivado Testbench Converter")
    print("=" * 40)
    
    test_cases = convert_testbench_to_vivado(str(input_file), str(output_file))
    
    print("=" * 40)
    print("Conversion Summary:")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Tests converted: {len(test_cases)}")
    for test in test_cases:
        print(f"  - {test}")
    print("\nThe converted file can now be used in Vivado simulation.")

if __name__ == "__main__":
    main()