from sly import Lexer, Parser  # Librería SLY para análisis léxico y sintáctico
import json  # Para cargar archivos JSON con instrucciones
import os    # Para manejo de rutas de archivos
import sys   # Para argumentos de línea de comandos

"""
RISC-V RV32I Assembler
This program converts RISC-V assembly code to machine code.
Supports:
- Basic RV32I instructions
- Common pseudoinstructions
- Labels for jumps
- Syntax and range validation
"""

# Definir el decorador _ para reglas de SLY
def _(pattern):
    """
    Decorator for SLY lexer and parser rules.
    Allows using @_('pattern') syntax instead of assigning func.pattern
    """
    def decorator(func):
        func.pattern = pattern
        return func
    return decorator

# ========================
#  ISA (Load JSONs with instruction definitions)
# ========================
# Get the base path of the current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load instruction definitions from JSON files
# Each file contains instructions of a specific type
ISA = {
    "R": json.load(open(os.path.join(base_dir, "Rtype.json"), encoding="utf-8")),  # Type R: Register-Register
    "I": json.load(open(os.path.join(base_dir, "Itype.json"), encoding="utf-8")),  # Type I: Immediate
    "S": json.load(open(os.path.join(base_dir, "Stype.json"), encoding="utf-8")),  # Type S: Store
    "B": json.load(open(os.path.join(base_dir, "Btype.json"), encoding="utf-8")),  # Type B: Branch
    "U": json.load(open(os.path.join(base_dir, "Utype.json"), encoding="utf-8")),  # Type U: Upper immediate
    "J": json.load(open(os.path.join(base_dir, "Jtype.json"), encoding="utf-8")),  # Type J: Jump
}

# Load pseudoinstruction definitions
# Pseudoinstructions expand to one or more real instructions
with open(os.path.join(base_dir, "pseudo.json"), encoding="utf-8") as f:
    PSEUDO_INSTRUCTIONS = json.load(f)

# Load register name to number mapping
# Allows using names like 'ra', 'sp', 't0' instead of x1, x2, x5
with open(os.path.join(base_dir, "REGnames.json"), encoding="utf-8") as f:
    REGnames = json.load(f)

# Create set of all valid mnemonics
# Includes both real instructions and pseudoinstructions
MNEMONICS = set()
for table in ISA.values():
    MNEMONICS.update(table.keys())
MNEMONICS.update(PSEUDO_INSTRUCTIONS.keys())

# ========================
#   PSEUDOINSTRUCTIONS
# ========================
def expand_pseudo_instruction(mnemonic, args):
    # Verificar si es una pseudoinstrucción válida
    if mnemonic not in PSEUDO_INSTRUCTIONS:
        return None
    
    # Obtener las plantillas de expansión
    templates = PSEUDO_INSTRUCTIONS[mnemonic]
    expanded = []
    
    # Convertir argumentos a strings para facilitar el reemplazo
    str_args = [str(arg) for arg in args]
    
    # Procesar cada plantilla de expansión
    for template in templates:
        instruction = template
        
        # Mapeo específico para diferentes tipos de pseudoinstrucciones
        if mnemonic in ["BEQZ", "BNEZ", "BLEZ", "BGEZ", "BLTZ", "BGTZ"]:
            # Instrucciones de branch con un solo registro comparado con cero
            if len(str_args) >= 1:
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{offset}", str_args[1])
        
        elif mnemonic in ["BGT", "BLE", "BGTU", "BLEU"]:
            # Instrucciones de branch comparando dos registros
            if len(str_args) >= 1:
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{rt}", f"x{str_args[1]}")
            if len(str_args) >= 3:
                instruction = instruction.replace("{offset}", str_args[2])
        
        elif mnemonic in ["LI", "LI_SMALL", "LI_LARGE"]:
            # Load immediate: load immediate value into register
            if len(str_args) >= 1:
                instruction = instruction.replace("{rd}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{imm}", str_args[1])
        
        elif mnemonic in ["MV", "NOT", "NEG", "SEQZ", "SNEZ", "SLTZ", "SGTZ"]:
            # Unary or move operations
            if len(str_args) >= 1:
                instruction = instruction.replace("{rd}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{rs}", f"x{str_args[1]}")
        
        elif mnemonic in ["J", "JAL"]:
            # Unconditional jumps with single operand (offset)
            if len(str_args) >= 1:
                instruction = instruction.replace("{offset}", str_args[0])
        
        elif mnemonic in ["JR", "JALR"]:
            # Jump to register with single operand (register)
            if len(str_args) >= 1:
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
        
        else:
            # Generic mapping for other pseudoinstructions
            if len(str_args) >= 1:
                instruction = instruction.replace("{rd}", f"x{str_args[0]}")
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{rs}", f"x{str_args[1]}")
                instruction = instruction.replace("{rt}", f"x{str_args[1]}")
                instruction = instruction.replace("{imm}", str_args[1])
                instruction = instruction.replace("{offset}", str_args[1])
            if len(str_args) >= 3:
                instruction = instruction.replace("{offset}", str_args[2])
                instruction = instruction.replace("{imm}", str_args[2])
        
        expanded.append(instruction)
    
    return expanded

# ========================
#  LEXER (Lexical Analysis)
# ========================
class RV32ILexer(Lexer):
    """
    Lexical analyzer for RISC-V RV32I assembly.
    Recognizes tokens like instructions, registers, numbers, etc.
    """
    # Define token types recognized by the lexer
    tokens = { 'INSTR', 'REG', 'NUMBER', 'COMMA', 'LPAREN', 'RPAREN', 'IDENT', 'COLON', 'DIRECTIVE', 'NEWLINE', 'STRING' }
    ignore = ' \t'  # Ignore spaces and tabs

    # Simple tokens (single character)
    COMMA  = r','     # Operand separator
    LPAREN = r'\('    # Left parenthesis for addresses
    RPAREN = r'\)'    # Right parenthesis for addresses
    COLON = r':'      # For label definitions

    @_(r'\.[A-Za-z]+')
    def DIRECTIVE(self, t):
        """
        Recognizes assembler directives (like .data, .text).
        Converts to lowercase for normalization.
        """
        t.value = t.value.lower()
        return t
    
    @_(r'-?(0x[0-9A-Fa-f]+|\d+)')
    def NUMBER(self, t):
        """
        Recognizes integer numbers in decimal or hexadecimal.
        Supports negative numbers.
        Automatically converts hex (0x...) and decimal to int.
        """
        if t.value.startswith("0x"):
            t.value = int(t.value, 16)  # Hexadecimal
        else:
            t.value = int(t.value)      # Decimal
        return t

    @_(r'"[^"]*"')
    def STRING(self, t):
        """
        Recognizes strings enclosed in double quotes.
        Removes quotes and keeps only the content.
        """
        t.value = t.value[1:-1]  # Remove quotes
        return t

    @_(r'x(?:[0-9]|[1-2][0-9]|3[0-1])\b|(?:zero|ra|sp|gp|tp|fp|t[0-6]|s(?:[0-9]|1[0-1])|a[0-7])\b')
    def REG(self, t):
        """
        Recognizes registers in x0-x31 format or symbolic names.
        Examples: x1, ra, sp, t0, s1, a0
        Converts symbolic names to numbers using REGnames.
        """
        v = t.value
        if v.startswith('x'):
            # Register in xN format
            t.value = int(v[1:])
        else:
            # Register with symbolic name
            t.value = REGnames[v]
        return t
    
    @_(r'[A-Za-z_][A-Za-z0-9_]*')
    def IDENT(self, t):
        """
        Recognizes identifiers (labels or instructions).
        If the identifier is a valid instruction, changes type to INSTR.
        """
        if t.value.upper() in MNEMONICS:
            t.type = 'INSTR'
            t.value = t.value.upper()  # Normalize instructions to uppercase
        else:
            t.value = t.value  # Keep as identifier (label)
        return t
    
    @_(r'\n+')
    def NEWLINE(self, t):
        """
        Recognizes newlines and updates line counter.
        """
        self.lineno += t.value.count('\n')
        return t

    @_(r'#.*')
    def COMMENT(self, t):
        pass  # Ignore comments

    def error(self, t):
        """
        Handles lexical errors when encountering unrecognized characters.
        """
        raise SyntaxError(f"Line {self.lineno}: illegal character {t.value[0]!r}")

# ========================
#  PARSER (Syntactic Analysis)  
# ========================
class AsmParser(Parser):
    """
    Syntactic analyzer for RISC-V RV32I assembly.
    Builds an abstract syntax tree (AST) of the assembly code.
    """
    tokens = RV32ILexer.tokens
    expected_shift_reduce = 1  # Configuration to resolve shift/reduce conflicts

    @_('statement_list')
    def program(self, p):
        """
        Main rule: a program is a list of statements.
        Filters out null statements (empty lines).
        """
        return [stmt for stmt in p.statement_list if stmt is not None]

    @_('statement')
    def statement_list(self, p):
        """
        Base case: list with a single statement.
        """
        return [p.statement]

    @_('statement_list statement')
    def statement_list(self, p):
        """
        Recursive case: add statement to existing list.
        """
        return p.statement_list + [p.statement]

    @_('declaration')
    def statement(self, p):
        """
        A statement can be an instruction or label.
        """
        return p.declaration
    
    @_('declaration NEWLINE')
    def statement(self, p):
        """
        A declaration followed by newline.
        """
        return p.declaration
    
    @_('NEWLINE')
    def statement(self, p):
        """
        Empty line (newline only).
        """
        return None

    @_('IDENT COLON')
    def declaration(self, p):
        """
        Label declaration: identifier followed by colon.
        """
        return ("LABEL", p.IDENT)

    @_('instruction')
    def declaration(self, p):
        """
        Instruction declaration.
        """
        return p.instruction
    
    @_('DIRECTIVE operand_list')
    def declaration(self, p):
        directive = p.DIRECTIVE
        # Extract only numeric values, identifiers, or strings
        values = []
        for op in p.operand_list:
            if op[0] == 'NUMBER':
                values.append(op[1])
            elif op[0] == 'IDENT':
                values.append(op[1])  # can be a label
            elif op[0] == 'STRING':
                values.append(op[1])  # string without quotes
            else:
                raise SyntaxError(f"Line {p.lineno}: invalid operand in {directive}")
        
        if directive in (".word", ".half", ".byte", ".string"):
            return ("DATA", directive, values)
        else:
            return ("DIRECTIVE", directive)

    @_('DIRECTIVE')
    def declaration(self, p):
        """
        Assembler directive declaration (.data, .text, etc.).
        """
        return ("DIRECTIVE", p.DIRECTIVE)

    @_('INSTR')
    def instruction(self, p):
        """
        Instruction without operands (e.g., EBREAK, ECALL).
        """
        self.current_line = p.lineno  # Store line number for errors
        instr = self.build_from_mnemonic(p.INSTR, [])
        if instr:
            instr.append(p.lineno)  # Add line number to instruction
            return tuple(instr)
        return None

    @_('INSTR operand_list')
    def instruction(self, p):
        """
        Instruction with operands.
        """
        self.current_line = p.lineno  # Store line number for errors
        instr = self.build_from_mnemonic(p.INSTR, p.operand_list)
        if instr:
            instr.append(p.lineno)  # Add line number to instruction
            return tuple(instr)
        return None

    @_('operand')
    def operand_list(self, p):
        """
        Operand list with single element.
        """
        return [p.operand]

    @_('operand COMMA operand_list')
    def operand_list(self, p):
        """
        Operand list with multiple elements separated by commas.
        """
        return [p.operand] + p.operand_list

    @_('NUMBER LPAREN REG RPAREN')
    def operand(self, p):
        """
        Memory operand: offset(register) - e.g., 100(x1)
        """
        return ('MEM', p.NUMBER, p.REG)

    @_('REG')
    def operand(self, p):
        """
        Register operand: x0, x1, ra, sp, etc.
        """
        return ('REG', p.REG)

    @_('NUMBER')
    def operand(self, p):
        """
        Immediate operand: integer number.
        """
        return ('NUMBER', p.NUMBER)

    @_('IDENT')
    def operand(self, p):
        """
        Identifier operand: label for jumps.
        """
        return ('IDENT', p.IDENT)

    @_('STRING')
    def operand(self, p):
        """
        String operand: quoted text for .string directive
        """
        return ('STRING', p.STRING)

    def build_from_mnemonic(self, mnemonic, operands):
        """
        Builds an internal representation of the instruction from mnemonic and operands.
        
        Args:
            mnemonic (str): Instruction name (e.g., "ADD", "LW", "BEQ")
            operands (list): List of parsed operands
        
        Returns:
            list: Internal representation of instruction with type, info and operands
        
        """
        line_num = getattr(self, 'current_line', 0)
        
        # Special handling for JALR which can be pseudoinstruction (1 op) or real instruction (2-3 ops)
        if mnemonic == "JALR":
            if len(operands) == 1:
                # It's a pseudoinstruction: jalr rs -> jalr x1, rs, 0
                if operands[0][0] == 'REG':
                    if not (0 <= operands[0][1] <= 31):
                        raise ValueError(f"Line {line_num}: Register x{operands[0][1]} invalid (range: x0-x31)")
                    args = [operands[0][1]]
                    return ["PSEUDO", mnemonic, args]
            # If it has 2 or 3 operands, continue as real instruction
        
        # Special handling for JAL which can be pseudoinstruction (1 op) or real instruction (2 ops)
        if mnemonic == "JAL":
            if len(operands) == 1:
                # It's a pseudoinstruction: jal offset -> jal x1, offset
                if operands[0][0] == 'IDENT':
                    args = [operands[0][1]]  # The label
                    return ["PSEUDO", mnemonic, args]
            # If it has 2 operands, continue as real instruction
        
        # Handle other pseudoinstructions (excluding JALR and JAL with more than 1 operand)
        if (mnemonic in PSEUDO_INSTRUCTIONS and 
            not (mnemonic == "JALR" and len(operands) > 1) and
            not (mnemonic == "JAL" and len(operands) > 1)):
            args = []
            # Extract and validate pseudoinstruction arguments
            for op in operands:
                if op[0] == 'REG':
                    if not (0 <= op[1] <= 31):
                        raise ValueError(f"Line {line_num}: Register x{op[1]} invalid (range: x0-x31)")
                    args.append(op[1])
                elif op[0] == 'NUMBER':
                    args.append(op[1])
                elif op[0] == 'IDENT':
                    args.append(op[1])
                elif op[0] == 'MEM':
                    if not (0 <= op[2] <= 31):
                        raise ValueError(f"Line {line_num}: Register x{op[2]} invalid (range: x0-x31)")
                    args.append((op[1], op[2]))
                else:
                    args.append(op)
            return ["PSEUDO", mnemonic, args]

        # Validate registers in basic operands
        for i, op in enumerate(operands):
            if op[0] == 'REG' and not (0 <= op[1] <= 31):
                raise ValueError(f"Line {line_num}: Register x{op[1]} invalid (range: x0-x31)")
            elif op[0] == 'MEM' and not (0 <= op[2] <= 31):
                raise ValueError(f"Line {line_num}: Register x{op[2]} invalid (range: x0-x31)")

        # Normalize operands to facilitate processing
        vals = []
        for op in operands:
            if op[0] == 'REG':
                vals.append(op[1])
            elif op[0] == 'NUMBER':
                vals.append(op[1])
            elif op[0] == 'IDENT':
                vals.append(op[1])
            elif op[0] == 'MEM':
                vals.append((op[1], op[2]))  # (offset, register)
            else:
                vals.append(op)

        # ===== R-TYPE INSTRUCTIONS =====
        # Format: op rd, rs1, rs2 (register-register-register)
        if mnemonic in ISA.get('R', {}):
            # Validate operand count
            if len(operands) != 3:
                raise SyntaxError(f"Line {line_num}: Instruction '{mnemonic}' requires 3 operands, found {len(operands)}")
            # Validate that all are registers
            for i, op in enumerate(operands):
                if op[0] != 'REG':
                    raise SyntaxError(f"Line {line_num}: Operand {i+1} of '{mnemonic}' must be register, found {op[0]}")
            
            if len(vals) == 3 and all(isinstance(v, int) for v in vals):
                info = ISA['R'][mnemonic]
                return ["R", info, vals[0], vals[1], vals[2]]  # [type, info, rd, rs1, rs2]

        # ===== I-TYPE INSTRUCTIONS =====
        # Format: op rd, rs1, imm OR op rd, offset(rs1) for loads
        if mnemonic in ISA.get('I', {}):
            # Special cases: EBREAK and ECALL have no operands
            if mnemonic in ["EBREAK", "ECALL"]:
                if len(operands) != 0:
                    raise SyntaxError(f"Line {line_num}: Instruction '{mnemonic}' should have no operands")
                info = ISA['I'][mnemonic]
                # Use the immediate defined in JSON (0 for ECALL, 1 for EBREAK)
                immediate_value = int(info[2], 2)
                return ["I", info, 0, 0, immediate_value]
    
            # Special validation for shift instructions
            if mnemonic in ["SLLI", "SRLI", "SRAI"]:
                shift = vals[2]
                if not (0 <= shift < 32):   # RV32 supports shifts 0-31
                    raise ValueError(f"invalid shift amount: {shift} in {mnemonic}")
    
            # Distinguish between normal format and load instructions
            if any(op[0] == 'MEM' for op in operands):
                # Load format: lw rd, offset(rs1)
                if len(operands) != 2:
                    raise SyntaxError(f"Line {line_num}: Instruction '{mnemonic}' (load) requires 2 operands, found {len(operands)}")
                if operands[0][0] != 'REG':
                    raise SyntaxError(f"Line {line_num}: First operand of '{mnemonic}' must be register")
                if operands[1][0] != 'MEM':
                    raise SyntaxError(f"Line {line_num}: Second operand of '{mnemonic}' must be offset(register)")
                
                if len(vals) == 2 and isinstance(vals[1], tuple):
                    rd = vals[0]
                    offset, rs1 = vals[1]
                    # Validate offset range for loads (-2048 to 2047)
                    if not (-2048 <= offset <= 2047):
                        raise ValueError(f"Line {line_num}: Offset {offset} out of range for load (-2048 to 2047)")
                    info = ISA['I'][mnemonic]
                    return ["I", info, rd, rs1, offset]
            else:
                # Normal format: addi rd, rs1, imm
                if len(operands) != 3:
                    raise SyntaxError(f"Line {line_num}: Instruction '{mnemonic}' requires 3 operands, found {len(operands)}")
                if operands[0][0] != 'REG':
                    raise SyntaxError(f"Line {line_num}: First operand of '{mnemonic}' must be register")
                if operands[1][0] != 'REG':
                    raise SyntaxError(f"Line {line_num}: Second operand of '{mnemonic}' must be register")
                if operands[2][0] != 'NUMBER':
                    raise SyntaxError(f"Line {line_num}: Third operand of '{mnemonic}' must be immediate")
                
                if len(vals) == 3:
                    rd, rs1, imm = vals[0], vals[1], vals[2]
                    # Validate immediate range (-2048 to 2047)
                    if not (-2048 <= imm <= 2047):
                        raise ValueError(f"Line {line_num}: Immediate {imm} out of range for I-type (-2048 to 2047)")
                    info = ISA['I'][mnemonic]
                    return ["I", info, rd, rs1, imm]

        # ===== S-TYPE INSTRUCTIONS =====
        # Format: sw rs2, offset(rs1) (store)
        if mnemonic in ISA.get('S', {}):
            if len(operands) != 2:
                raise SyntaxError(f"Line {line_num}: Instruction '{mnemonic}' requires 2 operands, found {len(operands)}")
            if operands[0][0] != 'REG':
                raise SyntaxError(f"Line {line_num}: First operand of '{mnemonic}' must be register")
            if operands[1][0] != 'MEM':
                raise SyntaxError(f"Line {line_num}: Second operand of '{mnemonic}' must be offset(register)")
            
            info = ISA['S'][mnemonic]
            if len(vals) == 2 and isinstance(vals[1], tuple):
                rs2 = vals[0]
                offset, rs1 = vals[1]
                # Validate offset range for stores (-2048 to 2047)
                if not (-2048 <= offset <= 2047):
                    raise ValueError(f"Line {line_num}: Offset {offset} out of range for store (-2048 to 2047)")
                return ["S", info, rs2, rs1, offset]

        # ===== B-TYPE INSTRUCTIONS =====
        # Format: beq rs1, rs2, label (branch)
        if mnemonic in ISA.get('B', {}):
            if len(operands) != 3:
                raise SyntaxError(f"Line {line_num}: Instruction '{mnemonic}' requires 3 operands, found {len(operands)}")
            if operands[0][0] != 'REG':
                raise SyntaxError(f"Line {line_num}: First operand of '{mnemonic}' must be register")
            if operands[1][0] != 'REG':
                raise SyntaxError(f"Line {line_num}: Second operand of '{mnemonic}' must be register")
            if operands[2][0] != 'IDENT':
                raise SyntaxError(f"Line {line_num}: Third operand of '{mnemonic}' must be label")
            
            if len(vals) == 3:
                info = ISA['B'][mnemonic]
                return ["B", info, vals[0], vals[1], vals[2]]  # [type, info, rs1, rs2, label]

        # ===== U-TYPE INSTRUCTIONS =====
        # Format: lui rd, imm (upper immediate)
        if mnemonic in ISA.get('U', {}):
            if len(operands) != 2:
                raise SyntaxError(f"Line {line_num}: Instruction '{mnemonic}' requires 2 operands, found {len(operands)}")
            if operands[0][0] != 'REG':
                raise SyntaxError(f"Line {line_num}: First operand of '{mnemonic}' must be register")
            if operands[1][0] != 'NUMBER':
                raise SyntaxError(f"Line {line_num}: Second operand of '{mnemonic}' must be immediate")
            
            if len(vals) == 2:
                rd, imm = vals[0], vals[1]
                # Validate immediate range (20 bits for U-type)
                if not (-524288 <= imm <= 524287):
                    raise ValueError(f"Line {line_num}: Immediate {imm} out of range for U-type (-524288 to 524287)")
                info = ISA['U'][mnemonic]
                return ["U", info, rd, imm]

        # ===== J-TYPE INSTRUCTIONS =====
        # Format: jal rd, label (jump and link)
        if mnemonic in ISA.get('J', {}):
            if len(operands) != 2:
                raise SyntaxError(f"Line {line_num}: Instruction '{mnemonic}' requires 2 operands, found {len(operands)}")
            if operands[0][0] != 'REG':
                raise SyntaxError(f"Line {line_num}: First operand of '{mnemonic}' must be register")
            if operands[1][0] != 'IDENT':
                raise SyntaxError(f"Line {line_num}: Second operand of '{mnemonic}' must be label")
            
            if len(vals) == 2:
                info = ISA['J'][mnemonic]
                return ["J", info, vals[0], vals[1]]  # [type, info, rd, label]

        # If we get here, the instruction could not be interpreted
        raise SyntaxError(f"Line {line_num}: Could not interpret instruction '{mnemonic}' with operands {operands}")

# ========================
#  FIRST PASS
# ========================
def first_pass(source_code):
    """
    First pass of the assembler: build label table and calculate addresses.
    Now handles .text and .data sections, with .word, .half, .byte, .string directives.
    """
    labels = {}                    # Label table
    instruction_addresses = {}     # Instruction addresses by line
    data_addresses = {}            # Data addresses by line
    
    PC_text = 0x00000000           # Counter for instructions
    PC_data = 0x00000000           # Counter for data (example base)
    section = ".text"              # Current section (default)
    
    for lineno, raw in enumerate(source_code.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        code = line
        # Handle labels
        if ':' in code:
            label, rest = code.split(':', 1)
            label = label.strip()
            if label:
                if section == ".text":
                    labels[label] = PC_text
                elif section == ".data":
                    labels[label] = PC_data
            code = rest.strip()

        if not code:
            continue

        # Change section
        if code.startswith(".text"):
            section = ".text"
            continue
        elif code.startswith(".data"):
            section = ".data"
            continue

        # Handle data directives
        if section == ".data":
            if code.startswith(".word"):
                valores = code.replace(".word", "").split(',')
                data_addresses[lineno] = PC_data
                PC_data += 4 * len(valores)
            elif code.startswith(".half"):
                valores = code.replace(".half", "").split(',')
                data_addresses[lineno] = PC_data
                PC_data += 2 * len(valores)
            elif code.startswith(".byte"):
                valores = code.replace(".byte", "").split(',')
                data_addresses[lineno] = PC_data
                PC_data += 1 * len(valores)
            elif code.startswith(".string"):
                # Extract the string (must be in quotes)
                string_part = code.replace(".string", "").strip()
                if string_part.startswith('"') and string_part.endswith('"'):
                    string_content = string_part[1:-1]  # Remove quotes
                    # Calculate bytes: string length + 1 for null terminator
                    string_bytes = len(string_content) + 1
                    data_addresses[lineno] = PC_data
                    PC_data += string_bytes
                else:
                    raise SyntaxError(f"Line {lineno}: String must be enclosed in double quotes")
            continue

        # Handle instructions
        if section == ".text":
            instruction_addresses[lineno] = PC_text
            PC_text += 4

    return labels, instruction_addresses, data_addresses, PC_text, PC_data

# ========================
#  SECOND PASS
# ========================
def assemble_instruction(instr, labels, instruction_addresses):
    """
    Converts a parsed instruction to 32-bit machine code.
    
    Args:
        instr (tuple): Parsed instruction with format [type, info, operands...]
        labels (dict): Label table for resolving jumps
        instruction_addresses (dict): Instruction addresses by line
    
    Returns:
        int: 32-bit word with machine code, or None if error
    
    """
    instr_type = instr[0]
    
    # Only process valid instructions
    if instr_type not in ["R", "I", "S", "B", "U", "J"]:
        return None

    line_num = instr[-1]  # Line number (last element)
    pc = instruction_addresses.get(line_num)  # Address of this instruction
    if pc is None:
        print(f"Advertencia: No se encontró PC para línea {line_num}")
        return None

    info = instr[1]  # Información de la instrucción desde JSON
    opcode = int(info[0], 2)  # Convertir opcode de binario a entero
    word = 0  # Palabra de máquina a construir

    # ===== INSTRUCCIONES TIPO R =====
    # Formato: funct7[31:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
    if instr_type == "R":
        rd, rs1, rs2 = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)  # Campo función 3 bits
        funct7 = int(info[2], 2)  # Campo función 7 bits
        word = (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

    # ===== INSTRUCCIONES TIPO I =====
    # Formato: imm[31:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
    elif instr_type == "I":
        rd, rs1, imm = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        
        # Determinar si es instrucción de shift basándose en el opcode y funct3
        is_shift_instruction = (
            len(info) > 2 and 
            info[0] == "0010011" and  # Opcode tipo I
            info[1] in ["001", "101"] and  # funct3 para SLLI/SRLI/SRAI
            info[2] in ["0000000", "0100000"]  # funct7 válido para shift
        )
        
        if is_shift_instruction:
            # Instrucciones de shift: formato especial con funct7 + shamt
            funct7 = int(info[2], 2)
            shamt = imm & 0x1F  # Solo 5 bits para shift amount (0-31)
            if not (0 <= imm <= 31):
                raise ValueError(f"Línea {line_num}: Shift amount {imm} debe estar entre 0-31")
            imm_12bit = (funct7 << 5) | shamt
        else:
            # Instrucciones I normales
            # Verificar rango válido para inmediatos tipo I
            if not (-2048 <= imm <= 2047):
                raise ValueError(f"Línea {line_num}: Inmediato {imm} fuera de rango para tipo I")
            
            # Manejar valores negativos en complemento a 2 para 12 bits
            if imm < 0:
                imm_12bit = (imm + 4096) & 0xFFF
            else:
                imm_12bit = imm & 0xFFF
        
        word = (imm_12bit << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

    # ===== INSTRUCCIONES TIPO B =====
    # Formato: imm[12|10:5] | rs2[24:20] | rs1[19:15] | funct3[14:12] | imm[4:1|11] | opcode[6:0]
    elif instr_type == "B":
        rs1, rs2, label = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        
        # Resolver etiqueta a dirección
        target_addr = labels.get(label)
        if target_addr is None:
            raise NameError(f"Línea {line_num}: Etiqueta '{label}' no definida")
        
        # Calcular offset relativo al PC actual
        offset = target_addr - pc
        
        # Verify offset is in range and even (multiple of 2)
        if not (-4096 <= offset <= 4094) or offset % 2 != 0:
            raise ValueError(f"Line {line_num}: Jump to '{label}' out of range")

        # Codificación especial del inmediato en tipo B (bits reordenados)
        imm12 = (offset >> 12) & 1      # Bit 12
        imm11 = (offset >> 11) & 1      # Bit 11  
        imm10_5 = (offset >> 5) & 0b111111  # Bits 10:5
        imm4_1 = (offset >> 1) & 0b1111     # Bits 4:1
        
        word = (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
               (funct3 << 12) | (imm4_1 << 8) | (imm11 << 7) | opcode
    
    # ===== INSTRUCCIONES TIPO J =====
    # Formato: imm[20|10:1|11|19:12] | rd[11:7] | opcode[6:0]
    elif instr_type == "J":
        rd, label = instr[2], instr[3]
        
        # Resolver etiqueta a dirección
        target_addr = labels.get(label)
        if target_addr is None:
            raise NameError(f"Línea {line_num}: Etiqueta '{label}' no definida")
        
        # Calcular offset relativo al PC actual
        offset = target_addr - pc
        
        # Verify valid range for JAL and that it's even
        if not (-1048576 <= offset <= 1048574) or offset % 2 != 0:
            raise ValueError(f"Line {line_num}: Jump to '{label}' out of range for JAL")

        # Codificación especial del inmediato en tipo J (bits reordenados)
        imm20 = (offset >> 20) & 1          # Bit 20
        imm19_12 = (offset >> 12) & 0xFF    # Bits 19:12
        imm11 = (offset >> 11) & 1          # Bit 11
        imm10_1 = (offset >> 1) & 0x3FF     # Bits 10:1
        
        word = (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12) | (rd << 7) | opcode
    
    # ===== INSTRUCCIONES TIPO U =====
    # Formato: imm[31:12] | rd[11:7] | opcode[6:0]
    elif instr_type == "U":
        rd, imm = instr[2], instr[3]
        
        # Verificar rango válido para inmediatos tipo U (20 bits)
        if not (-524288 <= imm <= 524287):
            raise ValueError(f"Línea {line_num}: Inmediato {imm} fuera de rango para tipo U")
        
        # El inmediato va en los bits superiores (31:12)
        word = ((imm & 0xFFFFF) << 12) | (rd << 7) | opcode
    
    # ===== INSTRUCCIONES TIPO S =====
    # Formato: imm[11:5] | rs2[24:20] | rs1[19:15] | funct3[14:12] | imm[4:0] | opcode[6:0]
    elif instr_type == "S":
        rs2, rs1, offset = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        
        # Verificar rango del offset
        if not (-2048 <= offset <= 2047):
            raise ValueError(f"Línea {line_num}: Offset {offset} fuera de rango para tipo S")
        
        # Dividir el inmediato en dos partes según el formato S
        imm11_5 = (offset >> 5) & 0x7F  # Bits superiores (11:5)
        imm4_0 = offset & 0x1F          # Bits inferiores (4:0)
        
        word = (imm11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm4_0 << 7) | opcode

    return word

def second_pass(instructions, labels, instruction_addresses, data_addresses):
    """
    Second pass of the assembler:
    - Converts instructions to machine code.
    - Converts data directives (.word, .half, .byte, .string) to data memory.
    
    Args:
        instructions (list): List of parsed instructions and directives.
        labels (dict): Tabla de etiquetas.
        instruction_addresses (dict): Direcciones de instrucciones.
        data_addresses (dict): Direcciones de datos.
    
    Returns:
        tuple: (machine_code, data_memory)
            - machine_code: lista de palabras de 32 bits (instrucciones)
            - data_memory: lista de valores de datos expandidos en memoria
    """
    machine_code = []
    data_memory = []

    for instr in instructions:
        if not instr:
            continue

        # === MANEJO DE DIRECTIVAS DE DATOS ===
        if instr[0] == "DATA":
            directive, values = instr[1], instr[2]
            if directive == ".word":
                for v in values:
                    data_memory.append(v & 0xFFFFFFFF)  # 32 bits
            elif directive == ".half":
                for v in values:
                    data_memory.append(v & 0xFFFF)      # 16 bits
            elif directive == ".byte":
                for v in values:
                    data_memory.append(v & 0xFF)        # 8 bits
            elif directive == ".string":
                # Para strings, values[0] es el string sin comillas
                string_content = values[0]
                for char in string_content:
                    data_memory.append(ord(char))       # Código ASCII de cada carácter
                data_memory.append(0)                   # Terminador null
            else:
                print(f"Advertencia: directiva {directive} no implementada")
            continue  # saltar a la siguiente

        # === MANEJO DE INSTRUCCIONES ===
        code_word = assemble_instruction(instr, labels, instruction_addresses)
        if code_word is not None:
            machine_code.append(code_word)

    return machine_code, data_memory


def expand_all_pseudo(instructions, lexer, parser):

    final_instructions = []
    if not instructions:
        return []
        
    for instr in instructions:
        if not instr: 
            continue
        
        # Si es una pseudoinstrucción, expandirla
        if instr[0] == "PSEUDO":
            mnemonic, args, lineno = instr[1], instr[2], instr[-1]
            expanded_lines = expand_pseudo_instruction(mnemonic, args)
            
            # Re-parsear cada instrucción expandida
            for line in expanded_lines:
                parsed_expanded = parser.parse(lexer.tokenize(line))
                if parsed_expanded:
                    new_instr_tuple = parsed_expanded[0]
                    if isinstance(new_instr_tuple, list):
                        new_instr_tuple = tuple(new_instr_tuple)
                    
                    # Mantener el número de línea original
                    new_instr_list = list(new_instr_tuple)
                    new_instr_list.append(lineno)
                    final_instructions.append(tuple(new_instr_list))
        else:
            # Si es una instrucción normal, mantenerla
            final_instructions.append(instr)
    return final_instructions

# ========================
#  MAIN FUNCTION
# ========================
def main():
    """
    Main function of the RISC-V RV32I assembler.
    
    Usage: python assembler.py program.asm program.hex program.bin
    
    Complete assembly process:
    1. Parse command line arguments
    2. Read assembly code file
    3. First pass: build label table
    4. Lexical and syntactic analysis
    5. Pseudoinstruction expansion
    6. Second pass: generate machine code
    7. Write output files (hex and binary)
    8. Display results on console
    """
    print("=== RV32I Assembler ===")
    
    # ===== STEP 1: PARSE COMMAND LINE ARGUMENTS =====
    if len(sys.argv) != 4:
        print("Usage: python assembler.py program.asm program.hex program.bin")
        print("  program.asm - Input assembly file")
        print("  program.hex - Output hexadecimal file")
        print("  program.bin - Output binary file")
        return
    
    input_file = sys.argv[1]    # Input .asm file
    output_hex = sys.argv[2]    # Output .hex file
    output_bin = sys.argv[3]    # Output .bin file
    
    print(f"Input file: {input_file}")
    print(f"Output hex: {output_hex}")
    print(f"Output bin: {output_bin}")
    
    # ===== STEP 2: READ INPUT FILE =====
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = f.read()
        print(f"File '{input_file}' read successfully")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # ===== STEP 3: INITIALIZE ANALYZERS =====
    lexer = RV32ILexer()    # Lexical analyzer
    parser = AsmParser()    # Syntactic analyzer
    
    try:
        # ===== STEP 4: FIRST PASS =====
        print("\n=== FIRST PASS ===")
        labels, instruction_addresses, data_addresses, PC_text, PC_data = first_pass(data)
        print(f"Labels found: {labels}")
        print(f"Instruction addresses (text): {instruction_addresses}")
        print(f"Data addresses: {data_addresses}")
        print(f"Final .text PC: 0x{PC_text:08X} ({PC_text} bytes)")
        print(f"Final .data PC: 0x{PC_data:08X} ({PC_data} bytes)")


        
        # ===== STEP 5: SYNTACTIC ANALYSIS =====
        print("\n=== PARSING ===")
        try:
            result = parser.parse(lexer.tokenize(data))
            if result is None:
                print("Error: Could not parse the file. Check syntax.")
                return
            parsed_instructions = list(result)
        except (SyntaxError, ValueError) as e:
            print(f"Validation error: {e}")
            print("Assembly stops due to errors in source code.")
            return
        except Exception as e:
            print(f"Unexpected error during parsing: {e}")
            return
        
        # ===== STEP 6: PSEUDOINSTRUCTION EXPANSION =====
        print("Expandiendo pseudoinstrucciones...")
        final_instructions = expand_all_pseudo(parsed_instructions, lexer, parser)
        print(f"Total de instrucciones después de expansión: {len(final_instructions)}")
        
        # ===== STEP 7: SECOND PASS =====
        # SEGUNDA PASADA
        print("\n=== SEGUNDA PASADA ===")
        machine_code, data_memory = second_pass(final_instructions, labels, instruction_addresses, data_addresses)

        print(f"Machine code generated: {len(machine_code)} instructions")
        print(f"Data in memory: {len(data_memory)} values")

        
        if not machine_code and not data_memory:
            print("Error: No machine code or data generated")
            return
        
        if not machine_code:
            print("Warning: Only data generated, no instructions")
        
        # ===== STEP 8: GENERATE OUTPUT FILES =====
        print("\n=== GENERATING FILES ===")

        # Data file (always generated if there's data)
        if data_memory:
            data_hex_file = output_hex.replace('.hex', '_data.hex')
            with open(data_hex_file, "w") as f:
                for word in data_memory:
                    f.write(f"{word:08x}\n")
            print(f"File '{data_hex_file}' generated")
            
            # Archivo de mapa de memoria de datos
            with open("memory_map.txt", "w") as f:
                f.write("=== MAPA DE MEMORIA .DATA ===\n")
                f.write("-" * 80 + "\n")
                
                # Extraer información de tipos de datos del código fuente
                data_info = {}
                section = ".text"
                for line in data.splitlines():
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Cambiar de sección
                    if line.startswith(".data"):
                        section = ".data"
                        continue
                    elif line.startswith(".text"):
                        section = ".text"
                        continue
                    
                    if section == ".data" and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            label = parts[0].strip()
                            directive_part = parts[1].strip()
                            
                            if directive_part.startswith('.word'):
                                data_info[label] = '.word'
                            elif directive_part.startswith('.half'):
                                data_info[label] = '.half'
                            elif directive_part.startswith('.byte'):
                                data_info[label] = '.byte'
                            elif directive_part.startswith('.string'):
                                data_info[label] = '.string'
                
                # Crear lista de etiquetas ordenadas por dirección
                data_labels = [(addr, label) for label, addr in labels.items() if addr >= 0x10000000]
                data_labels.sort()  # Ordenar por dirección
                
                memory_index = 0  # Índice que va recorriendo data_memory
                for addr, label in data_labels:
                    if memory_index < len(data_memory):
                        data_type = data_info.get(label, '.word')  # Default a .word si no se encuentra
                        
                        if data_type == '.string':
                            # Para strings, mostrar los caracteres como string readable
                            string_chars = []
                            j = memory_index
                            while j < len(data_memory) and data_memory[j] != 0:
                                string_chars.append(chr(data_memory[j]))
                                j += 1
                            string_content = ''.join(string_chars)
                            string_length = len(string_chars) + 1  # +1 para el terminador null
                            f.write(f"{label:<15}\t0x{addr:08x}\t\t{data_type}\t\t\"{string_content}\"\\0\t\t[{string_length} bytes]\n")
                            memory_index += string_length  # Avanzar el índice por todos los bytes del string
                        else:
                            # Para otros tipos, mostrar como antes
                            value = data_memory[memory_index]
                            f.write(f"{label:<15}\t0x{addr:08x}\t\t{data_type}\t\t0x{value:08x}\t\t{value}\n")
                            memory_index += 1  # Advance only 1 position for simple types
                
            print("File 'memory_map.txt' generated")
        
        # Code files (only if there are instructions)
        if machine_code:
            # Hexadecimal file (one word per line)
            with open(output_hex, "w") as f:
                for word in machine_code:
                    f.write(f"{word & 0xFFFFFFFF:08x}\n")
            print(f"File '{output_hex}' generated")
            
            # Binary file (32 bits per line in text format)
            with open(output_bin, "w") as f:
                for word in machine_code:
                    binary_str = f"{word & 0xFFFFFFFF:032b}"
                    f.write(binary_str + "\n")
            print(f"File '{output_bin}' generated")
        
        # ===== STEP 9: DISPLAY RESULTS =====
        if machine_code:
            print("\n=== CÓDIGO MÁQUINA ===")
            for i, word in enumerate(machine_code):
                pc_hex = f"{i*4:04x}"                    # Dirección en hex
                hex_word = f"{word & 0xFFFFFFFF:08x}"    # Palabra en hex
                bin_word = f"{word & 0xFFFFFFFF:032b}"   # Palabra en binario
                print(f"0x{pc_hex}: 0x{hex_word} | {bin_word}")
        
        if data_memory:
            print("\n=== DATOS EN MEMORIA ===")
            base_addr = 0x10000000  # Dirección base de .data
            for i, word in enumerate(data_memory):
                addr_hex = f"{base_addr + i:08x}"
                hex_word = f"{word:08x}"
                print(f"0x{addr_hex}: 0x{hex_word}")
        
        if machine_code:
            print(f"\nEnsamblado completado exitosamente!")
            print(f"Total: {len(machine_code)} instrucciones ({len(machine_code)*4} bytes)")
        else:
            print(f"\nProcesamiento de datos completado exitosamente!")
        
        if data_memory:
            print(f"Datos: {len(data_memory)} valores en memoria")
        
    except Exception as e:
        print(f"Error durante el ensamblado: {e}")
        import traceback
        traceback.print_exc()

# ===== PUNTO DE ENTRADA =====
if __name__ == "__main__":
    main()