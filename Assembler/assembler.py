from sly import Lexer, Parser
import json
import os
"""
*direcciones de carga valida
*.data 

"""

# Definir el decorador _ para reglas de SLY
def _(pattern):
    """Decorador para reglas de lexer y parser"""
    def decorator(func):
        func.pattern = pattern
        return func
    return decorator

# ========================
#  ISA (cargar JSONs)
# ========================
base_dir = os.path.dirname(os.path.abspath(__file__))
ISA = {
    "R": json.load(open(os.path.join(base_dir, "Rtype.json"), encoding="utf-8")),
    "I": json.load(open(os.path.join(base_dir, "Itype.json"), encoding="utf-8")),
    "S": json.load(open(os.path.join(base_dir, "Stype.json"), encoding="utf-8")),
    "B": json.load(open(os.path.join(base_dir, "Btype.json"), encoding="utf-8")),
    "U": json.load(open(os.path.join(base_dir, "Utype.json"), encoding="utf-8")),
    "J": json.load(open(os.path.join(base_dir, "Jtype.json"), encoding="utf-8")),
}

# Cargar pseudoinstrucciones
with open(os.path.join(base_dir, "pseudo.json"), encoding="utf-8") as f:
    PSEUDO_INSTRUCTIONS = json.load(f)

# Cargar nombres de registros
with open(os.path.join(base_dir, "REGnames.json"), encoding="utf-8") as f:
    REGnames = json.load(f)

# Crear conjunto de mnemonics incluyendo pseudoinstrucciones
MNEMONICS = set()
for table in ISA.values():
    MNEMONICS.update(table.keys())
MNEMONICS.update(PSEUDO_INSTRUCTIONS.keys())

# ========================
#  PSEUDOINSTRUCCIONES
# ========================
def expand_pseudo_instruction(mnemonic, args):
    if mnemonic not in PSEUDO_INSTRUCTIONS:
        return None
    
    templates = PSEUDO_INSTRUCTIONS[mnemonic]
    expanded = []
    
    # Convertir argumentos a strings para el reemplazo
    str_args = [str(arg) for arg in args]
    
    for template in templates:
        instruction = template
        
        # Mapeo específico para diferentes patrones de pseudoinstrucciones
        if mnemonic in ["BEQZ", "BNEZ", "BLEZ", "BGEZ", "BLTZ", "BGTZ"]:
            if len(str_args) >= 1:
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{offset}", str_args[1])
        
        elif mnemonic in ["BGT", "BLE", "BGTU", "BLEU"]:
            if len(str_args) >= 1:
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{rt}", f"x{str_args[1]}")
            if len(str_args) >= 3:
                instruction = instruction.replace("{offset}", str_args[2])
        
        elif mnemonic in ["LI", "LI_SMALL", "LI_LARGE"]:
            if len(str_args) >= 1:
                instruction = instruction.replace("{rd}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{imm}", str_args[1])
        
        elif mnemonic in ["MV", "NOT", "NEG", "SEQZ", "SNEZ", "SLTZ", "SGTZ"]:
            if len(str_args) >= 1:
                instruction = instruction.replace("{rd}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{rs}", f"x{str_args[1]}")
        
        elif mnemonic in ["J", "JAL_OFF"]:
            if len(str_args) >= 1:
                instruction = instruction.replace("{offset}", str_args[0])
        
        elif mnemonic in ["JR", "JALR_REG"]:
            if len(str_args) >= 1:
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
        
        else:
            # Mapeo genérico
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
#  LEXER
# ========================
class RV32ILexer(Lexer):
    tokens = { 'INSTR', 'REG', 'NUMBER', 'COMMA', 'LPAREN', 'RPAREN', 'IDENT', 'COLON', 'DIRECTIVE', 'NEWLINE' }
    ignore = ' \t'

    COMMA  = r','
    LPAREN = r'\('
    RPAREN = r'\)'
    COLON = r':'

    @_(r'\.[A-Za-z]+')
    def DIRECTIVE(self, t):
        t.value = t.value.lower()
        return t
    
    @_(r'-?(0x[0-9A-Fa-f]+|\d+)')
    def NUMBER(self, t):
        if t.value.startswith("0x"):
            t.value = int(t.value, 16)
        else:
            t.value = int(t.value)
        return t

    @_(r'x(?:[0-9]|[1-2][0-9]|3[0-1])\b|(?:zero|ra|sp|gp|tp|fp|t[0-6]|s(?:[0-9]|1[0-1])|a[0-7])\b')
    def REG(self, t):
        v = t.value
        if v.startswith('x'):
            t.value = int(v[1:])
        else:
            t.value = REGnames[v]
        return t
    
    @_(r'[A-Za-z_][A-Za-z0-9_]*')
    def IDENT(self, t):
        if t.value.upper() in MNEMONICS:
            t.type = 'INSTR'
            t.value = t.value.upper()
        else:
            t.value = t.value
        return t
    
    @_(r'\n+')
    def NEWLINE(self, t):
        self.lineno += t.value.count('\n')
        return t

    def error(self, t):
        raise SyntaxError(f"Línea {self.lineno}: caracter ilegal {t.value[0]!r}")

# ========================
#  PARSER
# ========================
class AsmParser(Parser):
    tokens = RV32ILexer.tokens
    expected_shift_reduce = 1

    @_('statement_list')
    def program(self, p):
        return [stmt for stmt in p.statement_list if stmt is not None]

    @_('statement')
    def statement_list(self, p):
        return [p.statement]

    @_('statement_list statement')
    def statement_list(self, p):
        return p.statement_list + [p.statement]

    @_('declaration')
    def statement(self, p):
        return p.declaration
    
    @_('declaration NEWLINE')
    def statement(self, p):
        return p.declaration
    
    @_('NEWLINE')
    def statement(self, p):
        return None

    @_('IDENT COLON')
    def declaration(self, p):
        return ("LABEL", p.IDENT)

    @_('instruction')
    def declaration(self, p):
        return p.instruction

    @_('DIRECTIVE')
    def declaration(self, p):
        return ("DIRECTIVE", p.DIRECTIVE)

    @_('INSTR')
    def instruction(self, p):
        self.current_line = p.lineno  # Guardar línea actual
        instr = self.build_from_mnemonic(p.INSTR, [])
        if instr:
            instr.append(p.lineno)
            return tuple(instr)
        return None

    @_('INSTR operand_list')
    def instruction(self, p):
        self.current_line = p.lineno  # Guardar línea actual
        instr = self.build_from_mnemonic(p.INSTR, p.operand_list)
        if instr:
            instr.append(p.lineno)
            return tuple(instr)
        return None

    @_('operand')
    def operand_list(self, p):
        return [p.operand]

    @_('operand COMMA operand_list')
    def operand_list(self, p):
        return [p.operand] + p.operand_list

    @_('NUMBER LPAREN REG RPAREN')
    def operand(self, p):
        return ('MEM', p.NUMBER, p.REG)

    @_('REG')
    def operand(self, p):
        return ('REG', p.REG)

    @_('NUMBER')
    def operand(self, p):
        return ('NUMBER', p.NUMBER)

    @_('IDENT')
    def operand(self, p):
        return ('IDENT', p.IDENT)

    def build_from_mnemonic(self, mnemonic, operands):
        line_num = getattr(self, 'current_line', 0)
        
        if mnemonic in PSEUDO_INSTRUCTIONS:
            args = []
            for op in operands:
                if op[0] == 'REG':
                    if not (0 <= op[1] <= 31):
                        raise ValueError(f"Línea {line_num}: Registro x{op[1]} no válido (rango: x0-x31)")
                    args.append(op[1])
                elif op[0] == 'NUMBER':
                    args.append(op[1])
                elif op[0] == 'IDENT':
                    args.append(op[1])
                elif op[0] == 'MEM':
                    if not (0 <= op[2] <= 31):
                        raise ValueError(f"Línea {line_num}: Registro x{op[2]} no válido (rango: x0-x31)")
                    args.append((op[1], op[2]))
                else:
                    args.append(op)
            return ["PSEUDO", mnemonic, args]

        # Validar operandos básicos primero
        for i, op in enumerate(operands):
            if op[0] == 'REG' and not (0 <= op[1] <= 31):
                raise ValueError(f"Línea {line_num}: Registro x{op[1]} no válido (rango: x0-x31)")
            elif op[0] == 'MEM' and not (0 <= op[2] <= 31):
                raise ValueError(f"Línea {line_num}: Registro x{op[2]} no válido (rango: x0-x31)")

        # Normalizar operandos
        vals = []
        for op in operands:
            if op[0] == 'REG':
                vals.append(op[1])
            elif op[0] == 'NUMBER':
                vals.append(op[1])
            elif op[0] == 'IDENT':
                vals.append(op[1])
            elif op[0] == 'MEM':
                vals.append((op[1], op[2]))
            else:
                vals.append(op)

        # Validación específica por tipo de instrucción
        if mnemonic in ISA.get('R', {}):
            # R-type: registro, registro, registro
            if len(operands) != 3:
                raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' requiere 3 operandos, encontrados {len(operands)}")
            for i, op in enumerate(operands):
                if op[0] != 'REG':
                    raise SyntaxError(f"Línea {line_num}: Operando {i+1} de '{mnemonic}' debe ser registro, encontrado {op[0]}")
            
            if len(vals) == 3 and all(isinstance(v, int) for v in vals):
                info = ISA['R'][mnemonic]
                return ["R", info, vals[0], vals[1], vals[2]]

        if mnemonic in ISA.get('I', {}):
            if mnemonic in ["EBREAK", "ECALL"]:
                # ebreak y ecall no necesitan operandos
                if len(operands) != 0:
                    raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' no debe tener operandos")
                info = ISA['I'][mnemonic]
                # Usar el immediate definido en el JSON para diferenciar ECALL (0) de EBREAK (1)
                immediate_value = int(info[2], 2)  # Convertir el immediate del JSON a entero
                return ["I", info, 0, 0, immediate_value]
    
            if mnemonic in ["SLLI", "SRLI", "SRAI"]:
                shift = vals[2]
                if not (0 <= shift < 32):   # para RV32
                    raise ValueError(f"corrimiento inválido: {shift} en {mnemonic}")
    
            # I-type puede ser: reg, reg, imm O reg, offset(reg) para loads
            if any(op[0] == 'MEM' for op in operands):
                # Formato load: lw rd, offset(rs1)
                if len(operands) != 2:
                    raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' (load) requiere 2 operandos, encontrados {len(operands)}")
                if operands[0][0] != 'REG':
                    raise SyntaxError(f"Línea {line_num}: Primer operando de '{mnemonic}' debe ser registro")
                if operands[1][0] != 'MEM':
                    raise SyntaxError(f"Línea {line_num}: Segundo operando de '{mnemonic}' debe ser offset(registro)")
                
                if len(vals) == 2 and isinstance(vals[1], tuple):
                    rd = vals[0]
                    offset, rs1 = vals[1]
                    # Validar rango del offset
                    if not (-2048 <= offset <= 2047):
                        raise ValueError(f"Línea {line_num}: Offset {offset} fuera de rango para load (-2048 a 2047)")
                    info = ISA['I'][mnemonic]
                    return ["I", info, rd, rs1, offset]
            else:
                # Formato normal: addi rd, rs1, imm
                if len(operands) != 3:
                    raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' requiere 3 operandos, encontrados {len(operands)}")
                if operands[0][0] != 'REG':
                    raise SyntaxError(f"Línea {line_num}: Primer operando de '{mnemonic}' debe ser registro")
                if operands[1][0] != 'REG':
                    raise SyntaxError(f"Línea {line_num}: Segundo operando de '{mnemonic}' debe ser registro")
                if operands[2][0] != 'NUMBER':
                    raise SyntaxError(f"Línea {line_num}: Tercer operando de '{mnemonic}' debe ser inmediato")
                
                if len(vals) == 3:
                    rd, rs1, imm = vals[0], vals[1], vals[2]
                    # Validar rango del inmediato
                    if not (-2048 <= imm <= 2047):
                        raise ValueError(f"Línea {line_num}: Inmediato {imm} fuera de rango para tipo I (-2048 a 2047)")
                    info = ISA['I'][mnemonic]
                    return ["I", info, rd, rs1, imm]


        if mnemonic in ISA.get('S', {}):
            # S-type: sw rs2, offset(rs1)
            if len(operands) != 2:
                raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' requiere 2 operandos, encontrados {len(operands)}")
            if operands[0][0] != 'REG':
                raise SyntaxError(f"Línea {line_num}: Primer operando de '{mnemonic}' debe ser registro")
            if operands[1][0] != 'MEM':
                raise SyntaxError(f"Línea {line_num}: Segundo operando de '{mnemonic}' debe ser offset(registro)")
            
            info = ISA['S'][mnemonic]
            if len(vals) == 2 and isinstance(vals[1], tuple):
                rs2 = vals[0]
                offset, rs1 = vals[1]
                # Validar rango del offset
                if not (-2048 <= offset <= 2047):
                    raise ValueError(f"Línea {line_num}: Offset {offset} fuera de rango para store (-2048 a 2047)")
                return ["S", info, rs2, rs1, offset]

        if mnemonic in ISA.get('B', {}):
            # B-type: beq rs1, rs2, label
            if len(operands) != 3:
                raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' requiere 3 operandos, encontrados {len(operands)}")
            if operands[0][0] != 'REG':
                raise SyntaxError(f"Línea {line_num}: Primer operando de '{mnemonic}' debe ser registro")
            if operands[1][0] != 'REG':
                raise SyntaxError(f"Línea {line_num}: Segundo operando de '{mnemonic}' debe ser registro")
            if operands[2][0] != 'IDENT':
                raise SyntaxError(f"Línea {line_num}: Tercer operando de '{mnemonic}' debe ser etiqueta")
            
            if len(vals) == 3:
                info = ISA['B'][mnemonic]
                return ["B", info, vals[0], vals[1], vals[2]]

        if mnemonic in ISA.get('U', {}):
            # U-type: lui rd, imm
            if len(operands) != 2:
                raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' requiere 2 operandos, encontrados {len(operands)}")
            if operands[0][0] != 'REG':
                raise SyntaxError(f"Línea {line_num}: Primer operando de '{mnemonic}' debe ser registro")
            if operands[1][0] != 'NUMBER':
                raise SyntaxError(f"Línea {line_num}: Segundo operando de '{mnemonic}' debe ser inmediato")
            
            if len(vals) == 2:
                rd, imm = vals[0], vals[1]
                # Validar rango del inmediato (20 bits)
                if not (-524288 <= imm <= 524287):
                    raise ValueError(f"Línea {line_num}: Inmediato {imm} fuera de rango para tipo U (-524288 a 524287)")
                info = ISA['U'][mnemonic]
                return ["U", info, rd, imm]

        if mnemonic in ISA.get('J', {}):
            # J-type: jal rd, label
            if len(operands) != 2:
                raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' requiere 2 operandos, encontrados {len(operands)}")
            if operands[0][0] != 'REG':
                raise SyntaxError(f"Línea {line_num}: Primer operando de '{mnemonic}' debe ser registro")
            if operands[1][0] != 'IDENT':
                raise SyntaxError(f"Línea {line_num}: Segundo operando de '{mnemonic}' debe ser etiqueta")
            
            if len(vals) == 2:
                info = ISA['J'][mnemonic]
                return ["J", info, vals[0], vals[1]]

        raise SyntaxError(f"Línea {line_num}: No se pudo interpretar instrucción '{mnemonic}' con operandos {operands}")

# ========================
#  PRIMERA PASADA
# ========================
def first_pass(source_code):
    """Primera pasada: construir tabla de etiquetas"""
    labels = {}
    instruction_addresses = {}
    PC = 0
    
    for lineno, raw in enumerate(source_code.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        code = line
        # Manejar etiquetas
        if ':' in code:
            label, rest = code.split(':', 1)
            label = label.strip()
            if label:
                if label in labels:
                    print(f"ADVERTENCIA: Etiqueta '{label}' redefinida en línea {lineno}")
                labels[label] = PC
            code = rest.strip()

        # Si hay una instrucción, incrementar PC
        if code:
            instruction_addresses[lineno] = PC
            PC += 4

    return labels, instruction_addresses, PC

# ========================
#  SEGUNDA PASADA
# ========================
def assemble_instruction(instr, labels, instruction_addresses):
    """Convierte una instrucción parseada a código máquina de 32 bits"""
    instr_type = instr[0]
    
    if instr_type not in ["R", "I", "S", "B", "U", "J"]:
        return None

    line_num = instr[-1]
    pc = instruction_addresses.get(line_num)
    if pc is None:
        print(f"Advertencia: No se encontró PC para línea {line_num}")
        return None

    info = instr[1]
    opcode = int(info[0], 2)
    word = 0

    if instr_type == "R":
        rd, rs1, rs2 = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        funct7 = int(info[2], 2)
        word = (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

    elif instr_type == "I":
        rd, rs1, imm = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        
        # Manejo especial para valores hexadecimales
        if imm == 0xff:
            imm = -1
        elif imm >= 0x80 and imm <= 0xff:
            imm = imm - 0x100
        elif imm >= 0x800:
            imm = imm - 0x1000
        
        if not (-2048 <= imm <= 2047):
            raise ValueError(f"Línea {line_num}: Inmediato {imm} fuera de rango para tipo I")
        
        imm_12bit = imm & 0xFFF
        word = (imm_12bit << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

    elif instr_type == "B":
        rs1, rs2, label = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        target_addr = labels.get(label)
        if target_addr is None:
            raise NameError(f"Línea {line_num}: Etiqueta '{label}' no definida")
        
        offset = target_addr - pc
        if not (-4096 <= offset <= 4094) or offset % 2 != 0:
            raise ValueError(f"Línea {line_num}: Salto a '{label}' fuera de rango")

        imm12 = (offset >> 12) & 1
        imm11 = (offset >> 11) & 1
        imm10_5 = (offset >> 5) & 0b111111
        imm4_1 = (offset >> 1) & 0b1111
        
        word = (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
               (funct3 << 12) | (imm4_1 << 8) | (imm11 << 7) | opcode
    
    elif instr_type == "J":
        rd, label = instr[2], instr[3]
        target_addr = labels.get(label)
        if target_addr is None:
            raise NameError(f"Línea {line_num}: Etiqueta '{label}' no definida")
        
        offset = target_addr - pc
        if not (-1048576 <= offset <= 1048574) or offset % 2 != 0:
            raise ValueError(f"Línea {line_num}: Salto a '{label}' fuera de rango para JAL")

        imm20 = (offset >> 20) & 1
        imm19_12 = (offset >> 12) & 0xFF
        imm11 = (offset >> 11) & 1
        imm10_1 = (offset >> 1) & 0x3FF
        word = (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12) | (rd << 7) | opcode
    
    elif instr_type == "U":
        rd, imm = instr[2], instr[3]
        if not (-524288 <= imm <= 524287):
            raise ValueError(f"Línea {line_num}: Inmediato {imm} fuera de rango para tipo U")
        word = ((imm & 0xFFFFF) << 12) | (rd << 7) | opcode
    
    elif instr_type == "S":
        rs2, rs1, offset = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        if not (-2048 <= offset <= 2047):
            raise ValueError(f"Línea {line_num}: Offset {offset} fuera de rango para tipo S")
        
        imm11_5 = (offset >> 5) & 0x7F
        imm4_0 = offset & 0x1F
        word = (imm11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm4_0 << 7) | opcode

    return word

def second_pass(instructions, labels, instruction_addresses):
    """Segunda pasada: generar código máquina"""
    machine_code = []
    for instr in instructions:
        if not instr: 
            continue
        code_word = assemble_instruction(instr, labels, instruction_addresses)
        if code_word is not None:
            machine_code.append(code_word)
    return machine_code

def expand_all_pseudo(instructions, lexer, parser):
    """Expandir todas las pseudoinstrucciones"""
    final_instructions = []
    if not instructions:
        return []
        
    for instr in instructions:
        if not instr: 
            continue
        
        if instr[0] == "PSEUDO":
            mnemonic, args, lineno = instr[1], instr[2], instr[-1]
            expanded_lines = expand_pseudo_instruction(mnemonic, args)
            
            for line in expanded_lines:
                parsed_expanded = parser.parse(lexer.tokenize(line))
                if parsed_expanded:
                    new_instr_tuple = parsed_expanded[0]
                    if isinstance(new_instr_tuple, list):
                        new_instr_tuple = tuple(new_instr_tuple)
                    
                    new_instr_list = list(new_instr_tuple)
                    new_instr_list.append(lineno)
                    final_instructions.append(tuple(new_instr_list))
        else:
            final_instructions.append(instr)
    return final_instructions

# ========================
#  FUNCIÓN PRINCIPAL
# ========================
def main():
    """Función principal del ensamblador"""
    print("=== Ensamblador RV32I ===")
    
    # Leer archivo de entrada (como el código simple)
    try:
        with open('arquitectura/Assembler/ejemplo.asm', 'r', encoding='utf-8') as f:
            data = f.read()
        print("Archivo 'ejemplo.asm' leído correctamente")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'ejemplo.asm'")
        return
    except Exception as e:
        print(f"Error leyendo archivo: {e}")
        return

    # Inicializar lexer y parser
    lexer = RV32ILexer()
    parser = AsmParser()
    
    try:
        # Primera pasada: buscar etiquetas y calcular direcciones
        print("\n=== PRIMERA PASADA ===")
        labels, instruction_addresses, final_pc = first_pass(data)
        print(f"Etiquetas encontradas: {labels}")
        print(f"PC final: 0x{final_pc:08X} ({final_pc} bytes)")
        
        # Parsear todas las instrucciones
        print("\n=== PARSING ===")
        try:
            result = parser.parse(lexer.tokenize(data))
            if result is None:
                print("Error: No se pudo parsear el archivo. Revisa la sintaxis.")
                return
            parsed_instructions = list(result)
        except (SyntaxError, ValueError) as e:
            print(f"Error de validación: {e}")
            print("El ensamblado se detiene debido a errores en el código fuente.")
            return
        except Exception as e:
            print(f"Error inesperado durante el parsing: {e}")
            return
        
        # Expandir pseudoinstrucciones
        print("Expandiendo pseudoinstrucciones...")
        final_instructions = expand_all_pseudo(parsed_instructions, lexer, parser)
        print(f"Total de instrucciones después de expansión: {len(final_instructions)}")
        
        # Segunda pasada: generar código máquina
        print("\n=== SEGUNDA PASADA ===")
        machine_code = second_pass(final_instructions, labels, instruction_addresses)
        
        if not machine_code:
            print("Error: No se generó código máquina")
            return
        
        print(f"Código máquina generado: {len(machine_code)} instrucciones")
        
        # Escribir archivos de salida (como el código simple)
        print("\n=== GENERANDO ARCHIVOS ===")
        
        # Archivo hexadecimal
        with open("output.hex", "w") as f:
            for word in machine_code:
                f.write(f"{word & 0xFFFFFFFF:08x}\n")
        print("Archivo 'output.hex' generado")
        
        # Archivo binario (texto con 0s y 1s)
        with open("output.bin", "w") as f:
            for word in machine_code:
                binary_str = f"{word & 0xFFFFFFFF:032b}"
                f.write(binary_str + "\n")
        print("Archivo 'output.bin' generado (formato texto binario)")
        
        # Mostrar resultado en consola
        print("\n=== CÓDIGO MÁQUINA ===")
        for i, word in enumerate(machine_code):
            pc_hex = f"{i*4:04x}"
            hex_word = f"{word & 0xFFFFFFFF:08x}"
            bin_word = f"{word & 0xFFFFFFFF:032b}"
            print(f"0x{pc_hex}: 0x{hex_word} | {bin_word}")
        
        print(f"\nEnsamblado completado exitosamente!")
        print(f"Total: {len(machine_code)} instrucciones ({len(machine_code)*4} bytes)")
        
    except Exception as e:
        print(f"Error durante el ensamblado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()