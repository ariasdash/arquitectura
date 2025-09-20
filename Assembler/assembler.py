from sly import Lexer, Parser
import json
import os

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


base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "REGnames.json"), encoding="utf-8") as f:
    contenido = f.read()
    REGnames = json.loads(contenido)

# Crear conjunto de mnemonics incluyendo pseudoinstrucciones
MNEMONICS = set()
for table in ISA.values():
    MNEMONICS.update(table.keys())
# Agregar pseudoinstrucciones a MNEMONICS
MNEMONICS.update(PSEUDO_INSTRUCTIONS.keys())

def expand_pseudo_instruction(mnemonic, args):
    if mnemonic not in PSEUDO_INSTRUCTIONS:
        return None
    
    templates = PSEUDO_INSTRUCTIONS[mnemonic]
    expanded = []
    
    # Convertir argumentos a strings para el reemplazo
    str_args = [str(arg) for arg in args]
    
    for template in templates:
        # Reemplazar los placeholders con los argumentos reales
        instruction = template
        
        # Mapeo específico para diferentes patrones de pseudoinstrucciones
        if mnemonic in ["BEQZ", "BNEZ", "BLEZ", "BGEZ", "BLTZ", "BGTZ"]:
            # Para branches: primer arg es rs, segundo es offset
            if len(str_args) >= 1:
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{offset}", str_args[1])
        
        elif mnemonic in ["BGT", "BLE", "BGTU", "BLEU"]:
            # Para branches comparativos: primer arg es rs, segundo es rt, tercero es offset
            if len(str_args) >= 1:
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{rt}", f"x{str_args[1]}")
            if len(str_args) >= 3:
                instruction = instruction.replace("{offset}", str_args[2])
        
        elif mnemonic in ["LI", "LI_SMALL", "LI_LARGE"]:
            # Para load immediate: primer arg es rd, segundo es imm
            if len(str_args) >= 1:
                instruction = instruction.replace("{rd}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{imm}", str_args[1])
        
        elif mnemonic in ["MV", "NOT", "NEG", "SEQZ", "SNEZ", "SLTZ", "SGTZ"]:
            # Para operaciones unarias: primer arg es rd, segundo es rs
            if len(str_args) >= 1:
                instruction = instruction.replace("{rd}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{rs}", f"x{str_args[1]}")
        
        elif mnemonic in ["J", "JAL_OFF"]:
            # Para jumps: primer arg es offset
            if len(str_args) >= 1:
                instruction = instruction.replace("{offset}", str_args[0])
        
        elif mnemonic in ["JR", "JALR_REG"]:
            # Para jump register: primer arg es rs
            if len(str_args) >= 1:
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
        
        elif mnemonic in ["CALL", "TAIL"]:
            # Para calls: primer arg es offset
            if len(str_args) >= 1:
                instruction = instruction.replace("{offset}", str_args[0])
        
        else:
            # Mapeo genérico para otros casos
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
    tokens = { 'INSTR', 'REG', 'NUMBER', 'COMMA', 'LPAREN', 'RPAREN', 'IDENT', 'COLON', 'DIRECTIVE' }
    ignore = ' \t'

    # Tokens simples
    COMMA  = r','
    LPAREN = r'\('
    RPAREN = r'\)'
    COLON = r':'

    # Directivas (.text, .data, .word, ...)
    @_(r'\.[A-Za-z]+')
    def DIRECTIVE(self, t):
        t.value = t.value.lower()
        return t
    
    # Números inmediatos (decimales o hexadecimales)
    @_(r'-?(0x[0-9A-Fa-f]+|\d+)')
    def NUMBER(self, t):
        if t.value.startswith("0x"):
            t.value = int(t.value, 16)
        else:
            t.value = int(t.value)
        return t

    #Registros: x0..x31 o nombres
    @_(r'x(?:[0-9]|[1-2][0-9]|3[0-1])\b|(?:zero|ra|sp|gp|tp|fp|t[0-6]|s(?:[0-9]|1[0-1])|a[0-7])\b')
    def REG(self, t):
        v = t.value
        if v.startswith('x'):
            t.value = int(v[1:])
        else:
            t.value = REGnames[v]   # REGnames debe mapear 'zero'->0, 'ra'->1, etc.
        return t
    
    # Identificadores (etiquetas o posibles mnemónicos)
    @_(r'[A-Za-z_][A-Za-z0-9_]*')
    def IDENT(self, t):
        # Si la palabra es un mnemónico conocido => token INSTR
        if t.value.upper() in MNEMONICS:
            t.type = 'INSTR'
            t.value = t.value.upper()
        else:
            # dejar IDENT en minúsculas facilita consistencia, pero opcional
            t.value = t.value
        return t
    
    # Nuevas líneas -> actualizar contador de línea
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        raise SyntaxError(f"Línea {self.lineno}: caracter ilegal {t.value[0]!r}")



# ========================
#  PARSER
# ========================
class AsmParser(Parser):
    tokens = RV32ILexer.tokens
    expected_shift_reduce = 1

    # Start rule: un programa es una lista de declaraciones
    @_('declaration_list')
    def program(self, p):
        return p.declaration_list

    # Lista de declaraciones
    @_('declaration')
    def declaration_list(self, p):
        return [p.declaration]

    @_('declaration declaration_list')
    def declaration_list(self, p):
        return [p.declaration] + p.declaration_list

    # Declaraciones posibles
    @_('IDENT COLON')
    def declaration(self, p):
        return ("LABEL", p.IDENT)

    @_('instruction')
    def declaration(self, p):
        return p.instruction

    @_('DIRECTIVE')
    def declaration(self, p):
        return ("DIRECTIVE", p.DIRECTIVE)

    # Connect all instruction types
    # Single unified instruction rule: INSTR optionally followed by a list of operands.
    @_('INSTR')
    def instruction(self, p):
        # Adjuntamos el número de línea a la instrucción parseada
        instr = self.build_from_mnemonic(p.INSTR, [])
        if instr:
            instr.append(p.lineno)
            return tuple(instr)
        return None

    @_('INSTR operand_list')
    def instruction(self, p):
        # Adjuntamos el número de línea a la instrucción parseada
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

    def build_r(self, info, rd, rs1, rs2):
        return ("R", info, rd, rs1, rs2)

    def build_i(self, info, rd, rs1, imm):
        return ("I", info, rd, rs1, imm)

    def build_b(self, info, rs1, rs2, offset):
        return ("B", info, rs1, rs2, offset)

    def build_s(self, info, rs2, rs1, offset):
        return ("S", info, rs2, rs1, offset)

    def build_u(self, info, rd, imm):
        return ("U", info, rd, imm)

    def build_j(self, info, rd, offset):
        return ("J", info, rd, offset)

    def build_pseudo(self, mnemonic, args):
        return ("PSEUDO", mnemonic, args)

    def build_from_mnemonic(self, mnemonic, operands):

        if mnemonic in PSEUDO_INSTRUCTIONS:
            # convertir operandos a lista simple (regs como enteros, numbers, idents)
            args = []
            for op in operands:
                if op[0] == 'REG':
                    args.append(op[1])
                elif op[0] == 'NUMBER':
                    args.append(op[1])
                elif op[0] == 'IDENT':
                    args.append(op[1])
                elif op[0] == 'MEM':
                    # Representar memoria como (imm, reg)
                    args.append((op[1], op[2]))
                else:
                    args.append(op)
            return ["PSEUDO", mnemonic, args]

        # No es pseudo: determinar tipo por tablas ISA y número/tipo de operandos
        # Normalizar operandos: extraer solo valores
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

        # Heurísticas simples para decidir el tipo (pueden ajustarse según ISA)
        if mnemonic in ISA.get('R', {}):
            if len(vals) == 3 and all(isinstance(v, int) for v in vals):
                info = ISA['R'][mnemonic]
                return ["R", info, vals[0], vals[1], vals[2]]

        if mnemonic in ISA.get('I', {}) and len(vals) == 3:
            info = ISA['I'][mnemonic]
            return ["I", info, vals[0], vals[1], vals[2]]

        if mnemonic in ISA.get('S', {}):
            info = ISA['S'][mnemonic]

            if len(vals) == 2:
                rs2 = vals[0]  # registro fuente
                if isinstance(vals[1], tuple) and len(vals[1]) == 2:
                    offset, rs1 = vals[1]  # offset y registro base de la memoria
                    return ["S", info, rs2, rs1, offset]
            # Formato alternativo: rs2, rs1, offset
            elif len(vals) == 3:
                return ["S", info, vals[0], vals[1], vals[2]]

        if mnemonic in ISA.get('B', {}) and len(vals) == 3:
            info = ISA['B'][mnemonic]
            return ["B", info, vals[0], vals[1], vals[2]]

        if mnemonic in ISA.get('U', {}) and len(vals) == 2:
            info = ISA['U'][mnemonic]
            return ["U", info, vals[0], vals[1]]

        if mnemonic in ISA.get('J', {}) and len(vals) == 2:
            info = ISA['J'][mnemonic]
            return ["J", info, vals[0], vals[1]]

        raise SyntaxError(f"No se pudo interpretar instrucción '{mnemonic}' con operandos {operands}")

def first_pass(source_code):
    labels = {}
    instruction_addresses = {}  # Mapea lineno -> PC
    PC = 0
    for lineno, raw in enumerate(source_code.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):  # Ignorar comentarios
            continue

        code = line
        # Una línea puede tener una etiqueta y una instrucción
        if ':' in code:
            label, rest = code.split(':', 1)
            label = label.strip()
            if label:
                if label in labels:
                    print(f"   ADVERTENCIA: Etiqueta '{label}' redefinida en línea {lineno}")
                labels[label] = PC
            code = rest.strip()

        if code:
            # Esta línea contiene una instrucción, guardar su PC y luego incrementarlo.
            instruction_addresses[lineno] = PC
            PC += 4

    return labels, instruction_addresses, PC

# ========================
#  SECOND PASS
# ========================

def assemble_instruction(instr, labels, instruction_addresses):
    """
    Convierte una única instrucción parseada a su código máquina de 32 bits.
    """
    instr_type = instr[0]
    
    if instr_type not in ["R", "I", "S", "B", "U", "J"]:
        return None

    line_num = instr[-1]
    pc = instruction_addresses.get(line_num)
    if pc is None:

        print(f"Advertencia: No se encontró PC para la instrucción en la línea (o derivada de la línea) {line_num}: {instr}")
        return None

    word = 0
    info = instr[1]
    
    # La info es una lista: [opcode, funct3, funct7] (para R/I/S/B) o [opcode] (para U/J)
    opcode = int(info[0], 2)

    if instr_type == "R":
        rd, rs1, rs2 = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        funct7 = int(info[2], 2)
        word = (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

    elif instr_type == "I":
        rd, rs1, imm = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        
        # Interpretación específica para valores hexadecimales de 8 bits
        # 0xff debe interpretarse como -1, no como +255
        if imm == 0xff:  # Caso específico: 0xff -> -1
            imm = -1
        elif imm >= 0x80 and imm <= 0xff:  # Valores 0x80-0xff como signed de 8 bits
            imm = imm - 0x100  # Convertir a complemento a 2 de 8 bits extendido
        elif imm >= 0x800:  # RISC-V estándar para valores >= 2048
            imm = imm - 0x1000  # Convertir a complemento a 2 de 12 bits
        
        # Verificar rango válido para RISC-V tipo I
        if not (-2048 <= imm <= 2047):
            raise ValueError(f"Línea {line_num}: Inmediato {imm} fuera de rango para tipo I (-2048 a 2047).")
        
        # Convertir a representación de 12 bits (complemento a 2 si es negativo)
        imm_12bit = imm & 0xFFF
        word = (imm_12bit << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

    elif instr_type == "B":
        rs1, rs2, label = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        target_addr = labels.get(label)
        if target_addr is None:
            raise NameError(f"Línea {line_num}: Etiqueta '{label}' no definida.")
        
        offset = target_addr - pc
        if not (-4096 <= offset <= 4094) or offset % 2 != 0:
            raise ValueError(f"Línea {line_num}: Salto a '{label}' fuera de rango ({offset} bytes).")

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
            raise NameError(f"Línea {line_num}: Etiqueta '{label}' no definida.")
        
        offset = target_addr - pc
        if not (-1048576 <= offset <= 1048574) or offset % 2 != 0:
            raise ValueError(f"Línea {line_num}: Salto a '{label}' fuera de rango para JAL.")

        imm20 = (offset >> 20) & 1
        imm19_12 = (offset >> 12) & 0xFF
        imm11 = (offset >> 11) & 1
        imm10_1 = (offset >> 1) & 0x3FF
        word = (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12) | (rd << 7) | opcode
    
    elif instr_type == "U":
        rd, imm = instr[2], instr[3]
        # Para U-type, el immediate ocupa los bits [31:12]
        if not (-524288 <= imm <= 524287):  # 20 bits signed
            raise ValueError(f"Línea {line_num}: Inmediato {imm} fuera de rango para tipo U.")
        word = ((imm & 0xFFFFF) << 12) | (rd << 7) | opcode
    
    elif instr_type == "S":
        rs2, rs1, offset = instr[2], instr[3], instr[4]
        funct3 = int(info[1], 2)
        if not (-2048 <= offset <= 2047):
            raise ValueError(f"Línea {line_num}: Offset {offset} fuera de rango para tipo S.")
        
        imm11_5 = (offset >> 5) & 0x7F  # bits [11:5]
        imm4_0 = offset & 0x1F          # bits [4:0]
        word = (imm11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm4_0 << 7) | opcode

    # Aquí iría la lógica para los tipos S y U
    # ...

    return word

def second_pass(instructions, labels, instruction_addresses):
    machine_code = []
    for instr in instructions:
        if not instr: continue
        code_word = assemble_instruction(instr, labels, instruction_addresses)
        if code_word is not None:
            machine_code.append(code_word)
    return machine_code

def write_binary_output(machine_code, filename="output.bin"):
    """Escribe el código máquina en un archivo binario."""
    with open(filename, "wb") as f:
        print(f"--- Escribiendo código máquina en '{filename}' ---")
        for i, word in enumerate(machine_code):
            # El PC de cada instrucción es 4 * i
            pc_hex = f"{i*4:04x}"
            # El código máquina se formatea a 32 bits (8 dígitos hexadecimales)
            hex_word = f"{word & 0xFFFFFFFF:08x}"
            print(f"Dirección 0x{pc_hex}: 0x{hex_word}")
            
            # Escribir la palabra de 32 bits en formato little-endian
            f.write(word.to_bytes(4, byteorder='little'))
    print(f"Archivo binario '{filename}' generado con {len(machine_code)} instrucciones ({len(machine_code)*4} bytes)")
    print("-------------------------------------------------")

def expand_all_pseudo(instructions, lexer, parser):
    final_instructions = []
    if not instructions:
        return []
        
    for instr in instructions:
        if not instr: continue
        
        if instr[0] == "PSEUDO":
            mnemonic, args, lineno = instr[1], instr[2], instr[-1]
            expanded_lines = expand_pseudo_instruction(mnemonic, args)
            
            for line in expanded_lines:
                # Parseamos cada línea expandida para obtener una instrucción real
                parsed_expanded = parser.parse(lexer.tokenize(line))
                if parsed_expanded:
                    new_instr_tuple = parsed_expanded[0]
                    
                    # Si es una lista, convertimos a tupla
                    if isinstance(new_instr_tuple, list):
                        new_instr_tuple = tuple(new_instr_tuple)
                    
                    # Convertimos a lista para poder añadir el lineno
                    new_instr_list = list(new_instr_tuple)
                    new_instr_list.append(lineno)
                    final_instructions.append(tuple(new_instr_list))
        else:
            final_instructions.append(instr)
    return final_instructions

def build_pseudo(mnemonic, args):
    expanded_instructions = expand_pseudo_instruction(mnemonic, args)
    if expanded_instructions is None:
        raise SyntaxError(f"Pseudoinstrucción no reconocida: {mnemonic}")
    return [("PSEUDO_EXPANDED", instr) for instr in expanded_instructions]

def test_assembler():
    source_code = """
    addi a1, a1, 5
    addi a2, a2, 0
    addi a3, a3, 1
    addi a4, a4, 0


    Loop:
    mv t1, a2
    mv a2, a3
    add a3, a3, t1
    sb t1, 0(t2)
    addi t2, t2, 4
    addi a4, a4, 1
    bltu a4, a1, Loop
    """
    lexer = RV32ILexer()
    parser = AsmParser()
    
    # --- Flujo de Ensamblado de Dos Pasadas ---

    # 1. Primera Pasada: Obtener direcciones de etiquetas e instrucciones
    labels, instruction_addresses, final_pc = first_pass(source_code)
    print("--- Primera Pasada ---")
    print("Labels:", labels)
    print("Instruction Addresses:", instruction_addresses)
    print(f"PC final estimado: 0x{final_pc:08X} ({final_pc} bytes)\n")

    # 2. Parseo del código fuente
    parsed_instructions = list(parser.parse(lexer.tokenize(source_code)))

    final_instructions = expand_all_pseudo(parsed_instructions, lexer, parser)
    print("--- Instrucciones Finales (post-expansión) ---")
    for instr in final_instructions:
        print(instr)
    print("")

 
    machine_code = second_pass(final_instructions, labels, instruction_addresses)
    
   
    print("--- Segunda Pasada (Código Máquina) ---")
    if not machine_code:
        print("No se generó código máquina.")
    else:
        for i, word in enumerate(machine_code):
            # El PC de cada instrucción es 4 * i
            pc_hex = f"{i*4:04x}"
            # El código máquina se formatea a 32 bits (8 dígitos hexadecimales)
            hex_word = f"{word & 0xFFFFFFFF:08x}"
            # Mostrar también en formato binario para referencia
            bin_word = f"{word & 0xFFFFFFFF:032b}"
            print(f"0x{pc_hex}: 0x{hex_word} | {bin_word}")
        
        print(f"\nTotal: {len(machine_code)} instrucciones ({len(machine_code)*4} bytes)")


if __name__ == "__main__":
    # Ejecutar test sencillo cuando se ejecute el archivo directamente
    test_assembler()