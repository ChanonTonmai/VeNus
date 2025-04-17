# Venus - DSL Interpreter for RISC-V CGRA FPGA

Venus is a Python-based DSL interpreter that converts domain-specific language code into YAML configuration for RISC-V CGRA FPGA. This tool simplifies the process of configuring and programming RISC-V-based Coarse-Grained Reconfigurable Arrays (CGRAs) on FPGAs.

## Features

- DSL to YAML conversion
- Support for RISC-V CGRA FPGA configuration
- Simple and intuitive syntax
- Extensible architecture

## Quick Start

### Running Venus

You can run Venus directly without installation:

```bash
# From the project root directory
python venus/cli/main.py input.dsl -o output.yaml
```

### Example Usage

1. Create a DSL file (e.g., `test.vn`):
```dsl
#pragma total_pes=4
int A[4][4] => x18, [i,j]
for i(0) = 0:4:
    load A[i][0]
endfor
```

2. Run Venus:
```bash
python venus/cli/main.py test.vn -o config.yaml
```

### DSL Syntax

The DSL (Domain-Specific Language) follows a simple syntax for defining CGRA configurations:

```dsl
# Configuration
#pragma total_pes=4
#pragma pes_per_cluster=2

# Memory declarations
int A[4][4] => x18, [i,j]

# Loops
for i(0) = 0:4:
    # Memory operations
    load A[i][0]
    store B[i][0]
    
    # Compute operations
    compute add x1 x2 x3
endfor
```

## Development

### Project Structure

```
venus/
├── core/           # Core compiler functionality
│   ├── __init__.py
│   └── compiler.py
├── cli/            # Command-line interface
│   ├── __init__.py
│   └── main.py
└── __init__.py
```

### Running Tests

```bash
# Run a specific test file
python venus/cli/main.py examples/conv_test.vn -o conv.yaml
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Author: K. Chanon
- Email: tonmai12369@gmail.com 