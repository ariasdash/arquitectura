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
    # Se conoce y aceptamos 1 conflicto shift/reduce debido al uso dual de
    # IDENT (etiquetas al inicio de línea) y operandos IDENT dentro de
    # instrucciones; es benigno en este diseño porque el parser maneja ambos
    # contextos correctamente. Definir expected_shift_reduce silencia el warning.
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
        return self.build_from_mnemonic(p.INSTR, [])

    @_('INSTR operand_list')
    def instruction(self, p):
        return self.build_from_mnemonic(p.INSTR, p.operand_list)

    # operand list: one or more operands separated by commas
    @_('operand')
    def operand_list(self, p):
        return [p.operand]

    @_('operand COMMA operand_list')
    def operand_list(self, p):
        return [p.operand] + p.operand_list

    # operand can be register, number, identifier (label) or memory offset like NUMBER LPAREN REG RPAREN
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

    # --- U-type and J-type merged ---
    # Esta regla se maneja ahora en pseudo_instr para evitar conflictos

    # Add these methods to your class:
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
        """Despacha la instrucción (mnemonic) y la lista de operandos a la forma
        correcta (R/I/S/B/U/J o PSEUDO). Esta función realiza la validación mínima
        y extrae valores (reg numbers, immediates, idents) para los builders.
        """
        # Si es pseudoinstrucción, convertir operandos a formas simples y devolver PSEUDO
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
            return self.build_pseudo(mnemonic, args)

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
                return self.build_r(info, vals[0], vals[1], vals[2])

        if mnemonic in ISA.get('I', {}) and len(vals) == 3:
            info = ISA['I'][mnemonic]
            return self.build_i(info, vals[0], vals[1], vals[2])

        if mnemonic in ISA.get('S', {}) and len(vals) == 3:
            info = ISA['S'][mnemonic]
            return self.build_s(info, vals[0], vals[1], vals[2])

        if mnemonic in ISA.get('B', {}) and len(vals) == 3:
            info = ISA['B'][mnemonic]
            return self.build_b(info, vals[0], vals[1], vals[2])

        if mnemonic in ISA.get('U', {}) and len(vals) == 2:
            info = ISA['U'][mnemonic]
            return self.build_u(info, vals[0], vals[1])

        if mnemonic in ISA.get('J', {}) and len(vals) == 2:
            info = ISA['J'][mnemonic]
            return self.build_j(info, vals[0], vals[1])

        # Si no se pudo casar, lanzar error claro
        raise SyntaxError(f"No se pudo interpretar instrucción '{mnemonic}' con operandos {operands}")


def first_pass(source_code):

    labels = {}
    PC = 0
    for lineno, raw in enumerate(source_code.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        code = line
        while True:
            if ':' in code:
                left, rest = code.split(':', 1)
                label = left.strip()
                if label:
                    if label in labels:
                        print(f"   ADVERTENCIA: Etiqueta '{label}' redefinida en línea {lineno}")
                    labels[label] = PC
                code = rest.strip()
                continue
            break


        if code:
            PC += 4

    return labels, PC

def build_pseudo(mnemonic, args):
    expanded_instructions = expand_pseudo_instruction(mnemonic, args)
    if expanded_instructions is None:
        raise SyntaxError(f"Pseudoinstrucción no reconocida: {mnemonic}")
    return [("PSEUDO_EXPANDED", instr) for instr in expanded_instructions]

def test_assembler():
    source_code = """
    start:  LI x1, 10
            LI x2, 20
            ADD x3, x1, x2
            BEQZ x3, end
            MV x4, x3
    end:    NOP
    """
    lexer = RV32ILexer()
    parser = AsmParser()
    
    # Primera pasada para etiquetas
    labels, final_pc = first_pass(source_code)
    print("Labels:", labels)
    print(f"PC final estimado: 0x{final_pc:08X} ({final_pc} bytes)")
    
    # Segunda pasada para parsear instrucciones
    instructions = parser.parse(lexer.tokenize(source_code))
    
    # Expandir pseudoinstrucciones
    final_instructions = []
    for instr in instructions:
        if instr[0] == "PSEUDO":
            mnemonic, args = instr[1], instr[2]
            expanded = build_pseudo(mnemonic, args)
            final_instructions.extend(expanded)
        else:
            final_instructions.append(instr)
    
    for instr in final_instructions:
        print(instr)


if __name__ == "__main__":
    # Ejecutar test sencillo cuando se ejecute el archivo directamente
    test_assembler()