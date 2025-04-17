#!/usr/bin/env python3

import argparse
import sys
import os
import yaml
import traceback

# Add parent directory to path so we can import the compiler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.compiler import DSLCompiler

def main():
    parser = argparse.ArgumentParser(description='DSL Interpreter for RISC-V CGRA FPGA')
    parser.add_argument('input_file', help='Path to the input DSL file')
    parser.add_argument('-o', '--output', help='Path to the output YAML file', default='output.yaml')
    
    args = parser.parse_args()
    
    try:
        print(f"Trying to open file: {args.input_file}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Check if file exists
        if not os.path.exists(args.input_file):
            print(f"Error: File '{args.input_file}' does not exist")
            sys.exit(1)
            
        # Read input file
        with open(args.input_file, 'r') as f:
            dsl_lines = f.readlines()
            print(f"Successfully read {len(dsl_lines)} lines from input file")
        
        # Initialize compiler
        compiler = DSLCompiler()
        
        # Compile DSL to YAML
        config = compiler.compile(dsl_lines)
        
        # Write output file
        with open(args.output, 'w') as f:
            class NoAliasDumper(yaml.Dumper):
                def ignore_aliases(self, data):
                    return True
            yaml.dump(config, f, sort_keys=False, default_flow_style=False, allow_unicode=True, Dumper=NoAliasDumper)
        
        print(f"Successfully compiled {args.input_file} to {args.output}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 