import yaml
import math
import re
from copy import deepcopy

class DSLCompiler:
    def __init__(self):
        self.mem_config = {f"x{i}": None for i in range(18, 26)}
        self.hardware_config = {
            "total_pes": 1,
            "clusters": {"count": 1, "pes_per_cluster": 1},
            "psrf_mem_offset": {f"x{i}_offset": None for i in range(18, 26)}
        }
        self.scheduling = {
            "minimum_pes_required": 1,
            "pe_assignments": []
        }
        self.loop_vars = []
        self.loop_hwl_map = {}
        self.variable_map = {}
        self.memory_base = {}
        self.emb_split = 1
        self.instruction_template = {}  # Initialize as empty dict
        self.hwl_index_base = 10
        self.loop_id_counter = 1
        self.current_pe = None
        self.functions = {}  # Store function definitions
        self.current_function = None  # Track current function being parsed
        self.function_addresses = {}  # Store function addresses
        self.next_function_address = 1000  # Start function addresses from 1000
        self.pc = 2  # Start PC at 2 (after HWL instructions)
        self.loop_pc_map = {}  # Map to store loop PC information
        self.loop_stack = []  # Stack to track nested loops
        self.nested_loop_pc = None  # Track PC for nested loops
        self.pe_pc = {}  # Dictionary to store PC for each PE


    def parse_pragma(self, line):
        if "total_pes" in line:
            self.hardware_config["total_pes"] = int(line.split("=")[1])
        elif "pes_per_cluster" in line:
            self.hardware_config["clusters"]["pes_per_cluster"] = int(line.split("=")[1])
        elif "EMB" in line:
            self.emb_split = int(line.split("=")[1])
        elif "minimum_pes_required" in line:
            self.scheduling["minimum_pes_required"] = int(line.split("=")[1])
            # Initialize PC for each PE after we know minimum_pes_required
            for pe_id in range(self.scheduling["minimum_pes_required"]):
                self.pe_pc[pe_id] = 0  # Start PC at 2 (after HWL instructions)

    def parse_memory(self, line):
        var, addr = line.replace("&", "").split("=")
        self.memory_base[var.strip()] = int(addr.strip())

    def parse_declaration(self, line):
        # Split the line into variable declaration and index specification
        parts = line.split(",", 1)
        decl_part = parts[0].strip()
        index_spec = parts[1].strip() if len(parts) > 1 else None
        
        # Parse the variable declaration part
        match = re.match(r"int\s+(\w+)\[((?:\d+\]\[)*\d+)\]\s*=>\s*(x\d+)", decl_part)
        if not match:
            raise ValueError(f"Invalid declaration format: {line}")
        
        varname, dim_str, reg = match.groups()
        dims = [int(d) for d in dim_str.split("][")]
        
        # Parse the index specification if present
        indices = []
        if index_spec:
            # Remove brackets and split by comma
            index_spec = index_spec.strip("[]")
            indices = [idx.strip() for idx in index_spec.split(",")]
            
            # Validate that number of indices matches number of dimensions
            if len(indices) != len(dims):
                raise ValueError(f"Number of indices ({len(indices)}) does not match number of dimensions ({len(dims)}) in {line}")
        
        # Store the variable information
        self.variable_map[varname] = {
            "shape": dims,
            "base_reg": reg,
            "var_id": len(self.variable_map),
            "indices": indices
        }
        
        print(f"Debug - Parsed variable {varname} with shape {dims} and indices {indices}")  # Debug print

    def parse_loop(self, line):
        # Remove 'for' and strip whitespace
        line = line.replace("for", "").strip()
        # Split on '(' to separate variable and register
        var_reg = line.split("(")[0].strip()
        # Split on '=' to get the count
        count_part = line.split("=")[1].strip().rstrip(":")
        count = int(count_part)
        # Get the register from between parentheses
        reg = line[line.find("(")+1:line.find(")")].strip()
        # Extract the number from the register name (e.g., 'x11' -> 11)
        hwl_index = int(reg[1:])  # Remove 'x' and convert to integer
        
        for pe_id in self.pe_pc:
            if self.pe_pc[pe_id] == 0:
                self.pe_pc[pe_id] += 1
            else:
                self.pe_pc[pe_id] += 2
        
        # Store current PC as loop start
        loop_start_pc = self.pe_pc[0]
        self.loop_pc_map[self.loop_id_counter] = {"start": loop_start_pc}
        
        # Create HWL instruction
        hwl_instr = {
            "operation": "HWL",
            "format": "hwl-type",
            "loop_id": self.loop_id_counter,
            "pc_start": loop_start_pc+1,
            "pc_stop": 0,  # Will be updated when we reach endfor
            "hwl_index": hwl_index,
            "iterations": count if len(self.loop_vars) > 0 else count // self.emb_split,
            "fill": -1
        }
        
        # Add to instruction template for all PEs
        for pe_id in range(self.scheduling["minimum_pes_required"]):
            if pe_id not in self.instruction_template:
                self.instruction_template[pe_id] = []
            self.instruction_template[pe_id].append(hwl_instr)
        
        # Update loop tracking
        self.loop_vars.append({"var": var_reg, "reg": reg, "iters": count, "loop_id": self.loop_id_counter, "hwl_index": hwl_index})
        self.loop_hwl_map[var_reg] = {"hwl_index": hwl_index, "loop_id": self.loop_id_counter, "iters": count}
        
        self.loop_stack.append(self.loop_id_counter)  # Push current loop_id to stack
        self.loop_id_counter += 1
        # Increment PC for HWL instruction for all PEs

        # Store the current PC as the start of this loop
        self.loop_pc_map[self.loop_id_counter] = self.pc
        # If this is a nested loop, store the PC for the outer loop
        if len(self.loop_stack) > 1:
            self.nested_loop_pc = self.pc

    def _parse_load(self, line, is_function=False):
        print(f"Debug - Parsing load instruction: '{line}'")  # Debug print
        indices_is_eq = False
        line = line.replace("load", "").strip()
        parts = line.split("[", 1)
        var = parts[0].strip()
        
        if len(parts) > 1:
            indices = [idx.strip("]") for idx in parts[1].split("][")]
            indices_has_eq = []
            index_groups = {}  # Dictionary to store index groups
            group_id = 0  # Counter for group IDs
            
            for idx in indices:
                if '+' in idx:
                    # Split on '+' and add both parts
                    parts = idx.split('+')
                    group_indices = [p.strip() for p in parts]
                    indices_has_eq.extend(group_indices)
                    indices_is_eq = True
                    
                    # Add all indices in this group to the same group
                    for group_idx in group_indices:
                        index_groups[group_idx] = group_id
                    group_id += 1
                else:
                    indices_has_eq.append(idx)
                    # Single index gets its own group
                    index_groups[idx] = group_id
                    group_id += 1
        else:
            indices = []
            indices_has_eq = []
            index_groups = {}
        
        if indices_is_eq:
            print(f"Debug - Load variable: {var}, indices: {indices_has_eq}, index groups: {index_groups}")  # Debug print
        else:
            print(f"Debug - Load variable: {var}, indices: {indices}")  # Debug print
        
        
        
        # For function parameters, use the parameter name directly
        if is_function:
            reg = var
        else:
            # For main program variables, use the base register
            reg = self.variable_map[var]["base_reg"] if var in self.variable_map else var
            
        if indices_is_eq:
            psrf = [self.loop_hwl_map[i]["hwl_index"] for i in indices_has_eq] + [0] * (6 - len(indices_has_eq))
            coeffs = self._calculate_coeffs(self.variable_map[var]["shape"], indices_has_eq, index_groups) if var in self.variable_map else [0] * 6
            coeffs += [0] * (6 - len(coeffs))
        else:
            psrf = [self.loop_hwl_map[i]["hwl_index"] for i in indices] + [0] * (6 - len(indices))
            coeffs = self._calculate_coeffs(self.variable_map[var]["shape"], indices, index_groups) if var in self.variable_map else [0] * 6
            coeffs += [0] * (6 - len(coeffs))   
        
        instruction = {
            "operation": "psrf.lw",
            "ra1": f"x{self.variable_map[var]['var_id'] + 1}" if var in self.variable_map else reg,
            "base_address": reg,
            "format": "psrf-mem-type",
            "var": self.variable_map[var]['var_id'] if var in self.variable_map else 0,
            "psrf_var": {f"v{i}": psrf[i] for i in range(6)},
            "coefficients": {f"c{i}": coeffs[i] for i in range(6)}
        }
        
        if is_function:
            print(f"Debug - Adding load instruction to function {self.current_function} for PE {self.current_pe}")  # Debug print
            self.functions[self.current_function]["instructions"][self.current_pe].append(instruction)
        else:
            print(f"Debug - Adding load instruction to main program for PE {self.current_pe}")  # Debug print
            self.instruction_template[self.current_pe].append(instruction)

    def _parse_store(self, line, is_function=False):
        print(f"Debug - Parsing store instruction: '{line}'")  # Debug print
        line = line.replace("store", "").strip()
        parts = line.split("[", 1)
        var = parts[0].strip()
        indices_is_eq = False

        # For function parameters, use the parameter name directly
        if is_function:
            reg = var
        else:
            # For main program variables, use the base register
            reg = self.variable_map[var]["base_reg"] if var in self.variable_map else var
            
        if len(parts) > 1:
            indices = [idx.strip("]") for idx in parts[1].split("][")]
            indices_has_eq = []
            index_groups = {}  # Dictionary to store index groups
            group_id = 0  # Counter for group IDs
            
            for idx in indices:
                if '+' in idx:
                    # Split on '+' and add both parts
                    parts = idx.split('+')
                    group_indices = [p.strip() for p in parts]
                    indices_has_eq.extend(group_indices)
                    indices_is_eq = True
                    
                    # Add all indices in this group to the same group
                    for group_idx in group_indices:
                        index_groups[group_idx] = group_id
                    group_id += 1
                else:
                    indices_has_eq.append(idx)
                    # Single index gets its own group
                    index_groups[idx] = group_id
                    group_id += 1
        else:
            indices = []
            indices_has_eq = []
            index_groups = {}
        
        if indices_is_eq:
            print(f"Debug - Store variable: {var}, indices: {indices_has_eq}, index groups: {index_groups}")  # Debug print
        else:
            print(f"Debug - Store variable: {var}, indices: {indices}")  # Debug print
        
        
        
        # For function parameters, use the parameter name directly
        if is_function:
            reg = var
        else:
            # For main program variables, use the base register
            reg = self.variable_map[var]["base_reg"] if var in self.variable_map else var
            
        if indices_is_eq:
            psrf = [self.loop_hwl_map[i]["hwl_index"] for i in indices_has_eq] + [0] * (6 - len(indices_has_eq))
            coeffs = self._calculate_coeffs(self.variable_map[var]["shape"], indices_has_eq, index_groups) if var in self.variable_map else [0] * 6
            coeffs += [0] * (6 - len(coeffs))
        else:
            psrf = [self.loop_hwl_map[i]["hwl_index"] for i in indices] + [0] * (6 - len(indices))
            coeffs = self._calculate_coeffs(self.variable_map[var]["shape"], indices, index_groups) if var in self.variable_map else [0] * 6
            coeffs += [0] * (6 - len(coeffs))   
        

        
        instruction = {
            "operation": "psrf.sw",
            "ra1": f"x{self.variable_map[var]['var_id'] + 1}" if var in self.variable_map else reg,
            "base_address": reg,
            "format": "psrf-mem-type",
            "var": self.variable_map[var]['var_id'] if var in self.variable_map else 0,
            "psrf_var": {f"v{i}": psrf[i] for i in range(6)},
            "coefficients": {f"c{i}": coeffs[i] for i in range(6)}
        }
        
        if is_function:
            print(f"Debug - Adding store instruction to function {self.current_function} for PE {self.current_pe}")  # Debug print
            self.functions[self.current_function]["instructions"][self.current_pe].append(instruction)
        else:
            print(f"Debug - Adding store instruction to main program for PE {self.current_pe}")  # Debug print
            self.instruction_template[self.current_pe].append(instruction)
    def _parse_nop(self, line):
        print(f"Debug - Parsing nop instruction: '{line}'")  # Debug print
        instruction = {
            "operation": "nop",
            "ra1": None,
            "ra2": None,
            "format": "nop-type"
        }
        self.instruction_template[self.current_pe].append(instruction)
    def _parse_compute(self, line, is_function=False):
        print(f"Debug - Parsing compute instruction: '{line}'")  # Debug print
        op, args = line.split(" ", 1)
        args = [arg.strip() for arg in args.split(",")]
        if len(args) != 3:
            raise ValueError(f"Invalid number of arguments for {op} operation: {line}. Expected 3 arguments (rd,ra1,ra2)")
        
        rd, ra1, ra2 = args
        instruction = {
            "operation": op.upper(),
            "rd": rd,
            "ra1": ra1,
            "ra2": ra2,
            "format": "r-type"
        }
        
        if is_function:
            print(f"Debug - Adding compute instruction to function {self.current_function} for PE {self.current_pe}")  # Debug print
            self.functions[self.current_function]["instructions"][self.current_pe].append(instruction)
        else:
            print(f"Debug - Adding compute instruction to main program for PE {self.current_pe}")  # Debug print
            self.instruction_template[self.current_pe].append(instruction)

    def _parse_function_call(self, line):
        # Parse function call like "call foo"
        parts = line.split()
        if len(parts) != 2 or parts[0] != "call":
            raise ValueError(f"Invalid function call format: {line}")
        
        func_name = parts[1]
        
        # Add function call instruction
        self.instruction_template[self.current_pe].append({
            "operation": "jal",
            "rd": "x26",  # Return address stored in x26
            "target": func_name,
            "format": "j-type",
            "address": self.function_addresses[func_name]  # Include function address
        })

    def _parse_goto(self, line, is_function=False):
        target = line.replace("goto", "").strip()
        instruction = {
            "operation": "jal",
            "rd": "x1",
            "target": target,
            "format": "j-type"
        }
        
        if is_function:
            self.functions[self.current_function]["instructions"][self.current_pe].append(instruction)
        else:
            self.instruction_template[self.current_pe].append(instruction)

    def _calculate_strides(self, shape):
        stride = 4
        strides = []
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        return strides
    
    def _calculate_coeffs(self, shape, indices, index_groups):
        # Initialize coefficients list with zeros
        coeffs = [0] * len(indices)
        
        # Get unique group IDs
        unique_groups = set(index_groups.values())
        
        # For each group, calculate its coefficient
        for group_id in unique_groups:
            # Get all indices in this group
            group_indices = [idx for idx, gid in index_groups.items() if gid == group_id]
            
            # Find the position of the first index in this group
            first_pos = indices.index(group_indices[0])
            
            # Calculate coefficient based on position
            # For position i, coefficient = product of shape[i+1:]
            coeff = 1
            for i in range(first_pos + 1, len(shape)):
                coeff *= shape[i]
            
            # Assign the same coefficient to all indices in this group
            for idx in group_indices:
                pos = indices.index(idx)
                coeffs[pos] = coeff
        
        return coeffs

    def build_yaml(self):
        cluster_count = self.hardware_config["total_pes"] // self.hardware_config["clusters"]["pes_per_cluster"]
        self.hardware_config["clusters"]["count"] = cluster_count

        # Set memory config
        for var, base in self.memory_base.items():
            reg = self.variable_map[var]["base_reg"]
            self.mem_config[reg] = base

        # Calculate offsets for variables using outermost loop
        if self.loop_vars:  # Check if we have any loop variables
            outer = self.loop_vars[0]
            outer_var = outer["var"]
            chunk_size = outer["iters"] // self.emb_split
            for var, meta in self.variable_map.items():
                shape = meta["shape"]
                indices = meta["indices"]
                if outer_var in ''.join(indices):
                    # Find the position of 'n' in indices
                    n_pos = indices.index('n') if 'n' in indices else -1
                    # Calculate stride by multiplying all shape elements except the one at n_pos
                    stride = 1
                    for i, dim in enumerate(shape):
                        if i != n_pos:
                            stride *= dim
                    reg = meta["base_reg"]
                    self.hardware_config["psrf_mem_offset"][f"{reg}_offset"] = chunk_size * stride


        pe_blocks = []
        for pe_id in range(self.scheduling["minimum_pes_required"]):
            # Create a copy of instructions without the fill field for HWL operations
            instructions = []
            for instr in self.instruction_template[pe_id]:
                if instr.get("operation") == "HWL":
                    # Create a new dict without the fill field
                    hwl_instr = {k: v for k, v in instr.items() if k != "fill"}
                    instructions.append(hwl_instr)
                else:
                    instructions.append(instr)
            
            pe_blocks.append({
                "pe_id": pe_id,
                "instructions": instructions
            })

        # Convert functions to YAML format with addresses and return instructions
        function_blocks = {}
        for func_name, func_data in self.functions.items():
            function_instructions = {}
            for pe_id, instrs in func_data["instructions"].items():
                # Add return instruction at the end of each function
                instrs.append({
                    "operation": "jalr",
                    "rd": "x0",
                    "ra1": "x26",
                    "imm": 0,
                    "format": "i-type"
                })
                function_instructions[pe_id] = instrs
            
            function_blocks[func_name] = {
                "address": self.function_addresses[func_name],
                "pe_assignments": [
                    {"pe_id": pe_id, "instructions": instrs}
                    for pe_id, instrs in function_instructions.items()
                ]
            }

        # Create the final YAML structure
        yaml_output = {
            "mem_config": self.mem_config,
            "hardware_config": self.hardware_config,
            "scheduling": {
                "minimum_pes_required": self.scheduling["minimum_pes_required"],
                "pe_assignments": pe_blocks
            }
        }

        # Add functions section if there are any functions
        if function_blocks:
            yaml_output["functions"] = function_blocks

        return yaml_output

    def compile(self, dsl_lines):
        # First pass: collect function definitions
        for line in dsl_lines:
            line = line.strip()
            if line.startswith("function"):
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid function definition format: {line}")
                
                func_name = parts[1]
                # Remove @ and : from address
                address_str = parts[2].replace("@", "").replace(":", "")
                try:
                    address = int(address_str)
                except ValueError:
                    raise ValueError(f"Invalid function address format: {parts[2]}")
                
                self.function_addresses[func_name] = address
                self.functions[func_name] = {
                    "instructions": {i: [] for i in range(2)}
                }

        # Second pass: process the rest of the code
        in_main = False
        for line in dsl_lines:
            line = line.strip()
            if not line:
                continue
            elif line.startswith("#"):
                self.parse_pragma(line)
                continue
            elif line == "main:":
                in_main = True
                print("Debug - Entering main section")
                # Initialize PC for all PEs
                for pe_id in range(self.scheduling["minimum_pes_required"]):
                    self.pe_pc[pe_id] = 0
            elif line == "endmain":
                in_main = False
                print("Debug - Exiting main section")
                # Increment PC for main section end for all PEs
                for pe_id in self.pe_pc:
                    self.pe_pc[pe_id] += 0
            elif line.startswith("int"):
                self.parse_declaration(line)
            elif line.startswith("&"):
                self.parse_memory(line)
            elif in_main:
                if line.startswith("&"):
                    self.parse_memory(line)
                elif line.startswith("int"):
                    self.parse_declaration(line)
                elif line.startswith("for"):
                    self.parse_loop(line)
                elif line == "endfor":
                    # End of loop section
                    print("Debug - End of loop section")  # Debug print
                    if self.loop_stack:
                        current_loop_id = self.loop_stack.pop()  # Get the most recent loop_id
                        # Update pc_stop for this loop in all PE instruction templates
                        for pe_id in range(self.scheduling["minimum_pes_required"]):
                            for instr in self.instruction_template[pe_id]:
                                if instr.get("operation") == "HWL" and instr.get("loop_id") == current_loop_id and instr.get("fill") == -1:
                                    instr["pc_stop"] = self.pe_pc[pe_id]
                                    instr["fill"] = 1
                                    print(f"Debug - Set pc_stop to {self.pe_pc[pe_id]} for loop_id {current_loop_id} in PE {pe_id}")
                        self.loop_id_counter -= 1
                        print(f"Debug - Decreased loop_id_counter to: {self.loop_id_counter}")
                    # Increment PC for endfor for all PEs
                    for pe_id in self.pe_pc:
                        self.pe_pc[pe_id] += 0
                elif line.startswith("pe_"):
                    # Extract PE number
                    self.current_pe = int(line.split("_")[1].split(":")[0])
                    print(f"Debug - Set current PE to: {self.current_pe}")  # Debug print
                    # Initialize instruction template and PC for this PE if not exists
                    if self.current_pe not in self.instruction_template:
                        self.instruction_template[self.current_pe] = []
                    if self.current_pe not in self.pe_pc:
                        self.pe_pc[self.current_pe] = 0
                elif line.startswith("function"):
                    # Start of function definition
                    parts = line.split()
                    if len(parts) < 3:
                        raise ValueError(f"Invalid function definition format: {line}")
                    
                    func_name = parts[1]
                    print(f"Debug - Found function definition: {func_name}")  # Debug print
                    self.current_function = func_name
                elif line == "end":
                    # End of function definition
                    print(f"Debug - End of function: {self.current_function}")  # Debug print
                    self.current_function = None
                elif line.startswith("end_pe"):
                    # End of PE 0 section
                    print("Debug - End of PE 0 section")  # Debug print
                elif self.current_pe is not None:
                    # Inside main program
                    print(f"Debug - Inside main program, PE: {self.current_pe}")  # Debug print
                    if line.startswith("load"):
                        self._parse_load(line)
                    elif line.startswith("store"):
                        self._parse_store(line)
                    elif line.startswith("mul") or line.startswith("add"):
                        self._parse_compute(line)
                    elif line.startswith("goto"):
                        self._parse_goto(line)
                    elif line.startswith("call"):  # Function call
                        self._parse_function_call(line)
                    elif line.startswith("nop"):
                        self._parse_nop(line)
                    # Increment PC only for the current PE
                    self.pe_pc[self.current_pe] += 1
                elif self.current_function is not None:
                    # Inside function definition
                    print(f"Debug - Inside function: {self.current_function}")  # Debug print
                    if line.startswith("pe_"):
                        self.current_pe = int(line.split("_")[1].split(":")[0])
                        print(f"Debug - Function PE set to: {self.current_pe}")  # Debug print
                        # Initialize instruction template for this PE if not exists
                        if self.current_pe not in self.instruction_template:
                            self.instruction_template[self.current_pe] = []
                    elif self.current_pe is not None:
                        if line.startswith("load"):
                            self._parse_load(line, is_function=True)
                        elif line.startswith("store"):
                            self._parse_store(line, is_function=True)
                        elif line.startswith("mul") or line.startswith("add"):
                            self._parse_compute(line, is_function=True)
                        elif line.startswith("goto"):
                            self._parse_goto(line, is_function=True)
                        elif line.startswith("nop"):
                            self._parse_nop(line)
                else:
                    # Inside main program
                    print(f"Debug - Inside main program, PE: {self.current_pe}")  # Debug print
                    if line.startswith("load"):
                        self._parse_load(line)
                    elif line.startswith("store"):
                        self._parse_store(line)
                    elif line.startswith("mul") or line.startswith("add"):
                        self._parse_compute(line)
                    elif line.startswith("goto"):
                        self._parse_goto(line)
                    elif line.startswith("call"):  # Function call
                        self._parse_function_call(line)
                    # Increment PC only for the current PE
                    self.pe_pc[self.current_pe] += 1

        return self.build_yaml()
    