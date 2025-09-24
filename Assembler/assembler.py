from sly import Lexer, Parser  # Librería SLY para análisis léxico y sintáctico
import json  # Para cargar archivos JSON con instrucciones
import os    # Para manejo de rutas de archivos

#directivas gpt 

"""
Ensamblador RISC-V RV32I
Este programa convierte código assembly RISC-V a código máquina.
Soporta:
- Instrucciones básicas RV32I
- Pseudoinstrucciones comunes
- Etiquetas para saltos
- Validación de sintaxis y rangos
"""

# Definir el decorador _ para reglas de SLY
def _(pattern):
    """
    Decorador para reglas de lexer y parser de SLY.
    Permite usar la sintaxis @_('pattern') en lugar de asignar func.pattern
    """
    def decorator(func):
        func.pattern = pattern
        return func
    return decorator

# ========================
#  ISA (Cargar JSONs con definiciones de instrucciones)
# ========================
# Obtener la ruta base del directorio actual
base_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar las definiciones de instrucciones desde archivos JSON
# Cada archivo contiene las instrucciones de un tipo específico
ISA = {
    "R": json.load(open(os.path.join(base_dir, "Rtype.json"), encoding="utf-8")),  # Tipo R: Registro-Registro
    "I": json.load(open(os.path.join(base_dir, "Itype.json"), encoding="utf-8")),  # Tipo I: Inmediato
    "S": json.load(open(os.path.join(base_dir, "Stype.json"), encoding="utf-8")),  # Tipo S: Store
    "B": json.load(open(os.path.join(base_dir, "Btype.json"), encoding="utf-8")),  # Tipo B: Branch
    "U": json.load(open(os.path.join(base_dir, "Utype.json"), encoding="utf-8")),  # Tipo U: Upper immediate
    "J": json.load(open(os.path.join(base_dir, "Jtype.json"), encoding="utf-8")),  # Tipo J: Jump
}

# Cargar definiciones de pseudoinstrucciones
# Las pseudoinstrucciones se expanden a una o más instrucciones reales
with open(os.path.join(base_dir, "pseudo.json"), encoding="utf-8") as f:
    PSEUDO_INSTRUCTIONS = json.load(f)

# Cargar mapeo de nombres de registros a números
# Permite usar nombres como 'ra', 'sp', 't0' en lugar de x1, x2, x5
with open(os.path.join(base_dir, "REGnames.json"), encoding="utf-8") as f:
    REGnames = json.load(f)

# Crear conjunto de todos los mnemónicos válidos
# Incluye tanto instrucciones reales como pseudoinstrucciones
MNEMONICS = set()
for table in ISA.values():
    MNEMONICS.update(table.keys())
MNEMONICS.update(PSEUDO_INSTRUCTIONS.keys())

# ========================
#  PSEUDOINSTRUCCIONES
# ========================
def expand_pseudo_instruction(mnemonic, args):
    """
    Expande una pseudoinstrucción a una o más instrucciones reales.
    
    Args:
        mnemonic (str): Nombre de la pseudoinstrucción (ej: "LI", "MV")
        args (list): Lista de argumentos de la pseudoinstrucción
    
    Returns:
        list: Lista de instrucciones expandidas en formato string, o None si no es pseudoinstrucción
    
    Ejemplos:
        LI x1, 100 -> ADDI x1, x0, 100
        MV x1, x2  -> ADDI x1, x2, 0
        BEQZ x1, label -> BEQ x1, x0, label
    """
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
            # Load immediate: cargar valor inmediato en registro
            if len(str_args) >= 1:
                instruction = instruction.replace("{rd}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{imm}", str_args[1])
        
        elif mnemonic in ["MV", "NOT", "NEG", "SEQZ", "SNEZ", "SLTZ", "SGTZ"]:
            # Operaciones unarias o de movimiento
            if len(str_args) >= 1:
                instruction = instruction.replace("{rd}", f"x{str_args[0]}")
            if len(str_args) >= 2:
                instruction = instruction.replace("{rs}", f"x{str_args[1]}")
        
        elif mnemonic in ["J", "JAL"]:
            # Saltos incondicionales con un solo operando (offset)
            if len(str_args) >= 1:
                instruction = instruction.replace("{offset}", str_args[0])
        
        elif mnemonic in ["JR", "JALR"]:
            # Saltos a registro con un solo operando (registro)
            if len(str_args) >= 1:
                instruction = instruction.replace("{rs}", f"x{str_args[0]}")
        
        else:
            # Mapeo genérico para otras pseudoinstrucciones
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
#  LEXER (Análisis Léxico)
# ========================
class RV32ILexer(Lexer):
    """
    Analizador léxico para assembly RISC-V RV32I.
    Reconoce tokens como instrucciones, registros, números, etc.
    """
    # Definir tipos de tokens que reconoce el lexer
    tokens = { 'INSTR', 'REG', 'NUMBER', 'COMMA', 'LPAREN', 'RPAREN', 'IDENT', 'COLON', 'DIRECTIVE', 'NEWLINE' }
    ignore = ' \t'  # Ignorar espacios y tabs

    # Tokens simples (un solo carácter)
    COMMA  = r','     # Separador de operandos
    LPAREN = r'\('    # Paréntesis izquierdo para direcciones
    RPAREN = r'\)'    # Paréntesis derecho para direcciones
    COLON = r':'      # Para definir etiquetas

    @_(r'\.[A-Za-z]+')
    def DIRECTIVE(self, t):
        """
        Reconoce directivas del ensamblador (como .data, .text).
        Convierte a minúsculas para normalizar.
        """
        t.value = t.value.lower()
        return t
    
    @_(r'-?(0x[0-9A-Fa-f]+|\d+)')
    def NUMBER(self, t):
        """
        Reconoce números enteros en decimal o hexadecimal.
        Soporta números negativos.
        Convierte automáticamente hex (0x...) y decimal a int.
        """
        if t.value.startswith("0x"):
            t.value = int(t.value, 16)  # Hexadecimal
        else:
            t.value = int(t.value)      # Decimal
        return t

    @_(r'x(?:[0-9]|[1-2][0-9]|3[0-1])\b|(?:zero|ra|sp|gp|tp|fp|t[0-6]|s(?:[0-9]|1[0-1])|a[0-7])\b')
    def REG(self, t):
        """
        Reconoce registros en formato x0-x31 o nombres simbólicos.
        Ejemplos: x1, ra, sp, t0, s1, a0
        Convierte nombres simbólicos a números usando REGnames.
        """
        v = t.value
        if v.startswith('x'):
            # Registro en formato xN
            t.value = int(v[1:])
        else:
            # Registro con nombre simbólico
            t.value = REGnames[v]
        return t
    
    @_(r'[A-Za-z_][A-Za-z0-9_]*')
    def IDENT(self, t):
        """
        Reconoce identificadores (etiquetas o instrucciones).
        Si el identificador es una instrucción válida, cambia el tipo a INSTR.
        """
        if t.value.upper() in MNEMONICS:
            t.type = 'INSTR'
            t.value = t.value.upper()  # Normalizar instrucciones a mayúsculas
        else:
            t.value = t.value  # Mantener como identificador (etiqueta)
        return t
    
    @_(r'\n+')
    def NEWLINE(self, t):
        """
        Reconoce saltos de línea y actualiza el contador de líneas.
        """
        self.lineno += t.value.count('\n')
        return t

    @_(r'#.*')
    def COMMENT(self, t):
        pass  # Ignore comments

    def error(self, t):
        """
        Maneja errores léxicos cuando encuentra caracteres no reconocidos.
        """
        raise SyntaxError(f"Línea {self.lineno}: caracter ilegal {t.value[0]!r}")

# ========================
#  PARSER (Análisis Sintáctico)
# ========================
class AsmParser(Parser):
    """
    Analizador sintáctico para assembly RISC-V RV32I.
    Construye un árbol de sintaxis abstracta (AST) del código assembly.
    """
    tokens = RV32ILexer.tokens
    expected_shift_reduce = 1  # Configuración para resolver conflictos shift/reduce

    @_('statement_list')
    def program(self, p):
        """
        Regla principal: un programa es una lista de declaraciones.
        Filtra declaraciones nulas (líneas vacías).
        """
        return [stmt for stmt in p.statement_list if stmt is not None]

    @_('statement')
    def statement_list(self, p):
        """
        Caso base: lista con una sola declaración.
        """
        return [p.statement]

    @_('statement_list statement')
    def statement_list(self, p):
        """
        Caso recursivo: agregar declaración a la lista existente.
        """
        return p.statement_list + [p.statement]

    @_('declaration')
    def statement(self, p):
        """
        Una declaración puede ser una instrucción o etiqueta.
        """
        return p.declaration
    
    @_('declaration NEWLINE')
    def statement(self, p):
        """
        Una declaración seguida de salto de línea.
        """
        return p.declaration
    
    @_('NEWLINE')
    def statement(self, p):
        """
        Línea vacía (solo salto de línea).
        """
        return None

    @_('IDENT COLON')
    def declaration(self, p):
        """
        Declaración de etiqueta: identificador seguido de dos puntos.
        """
        return ("LABEL", p.IDENT)

    @_('instruction')
    def declaration(self, p):
        """
        Declaración de instrucción.
        """
        return p.instruction
    
    @_('DIRECTIVE operand_list')
    def declaration(self, p):
        directive = p.DIRECTIVE
        # Extraer solo los valores numéricos o identificadores
        values = []
        for op in p.operand_list:
            if op[0] == 'NUMBER':
                values.append(op[1])
            elif op[0] == 'IDENT':
                values.append(op[1])  # puede ser etiqueta
            else:
                raise SyntaxError(f"Línea {p.lineno}: operando no válido en {directive}")
        
        if directive in (".word", ".half", ".byte"):
            return ("DATA", directive, values)
        else:
            return ("DIRECTIVE", directive)

    @_('DIRECTIVE')
    def declaration(self, p):
        """
        Declaración de directiva del ensamblador (.data, .text, etc.).
        """
        return ("DIRECTIVE", p.DIRECTIVE)

    @_('INSTR')
    def instruction(self, p):
        """
        Instrucción sin operandos (ej: EBREAK, ECALL).
        """
        self.current_line = p.lineno  # Guardar número de línea para errores
        instr = self.build_from_mnemonic(p.INSTR, [])
        if instr:
            instr.append(p.lineno)  # Agregar número de línea a la instrucción
            return tuple(instr)
        return None

    @_('INSTR operand_list')
    def instruction(self, p):
        """
        Instrucción con operandos.
        """
        self.current_line = p.lineno  # Guardar número de línea para errores
        instr = self.build_from_mnemonic(p.INSTR, p.operand_list)
        if instr:
            instr.append(p.lineno)  # Agregar número de línea a la instrucción
            return tuple(instr)
        return None

    @_('operand')
    def operand_list(self, p):
        """
        Lista de operandos con un solo elemento.
        """
        return [p.operand]

    @_('operand COMMA operand_list')
    def operand_list(self, p):
        """
        Lista de operandos con múltiples elementos separados por comas.
        """
        return [p.operand] + p.operand_list

    @_('NUMBER LPAREN REG RPAREN')
    def operand(self, p):
        """
        Operando de memoria: offset(registro) - ej: 100(x1)
        """
        return ('MEM', p.NUMBER, p.REG)

    @_('REG')
    def operand(self, p):
        """
        Operando de registro: x0, x1, ra, sp, etc.
        """
        return ('REG', p.REG)

    @_('NUMBER')
    def operand(self, p):
        """
        Operando inmediato: número entero.
        """
        return ('NUMBER', p.NUMBER)

    @_('IDENT')
    def operand(self, p):
        """
        Operando identificador: etiqueta para saltos.
        """
        return ('IDENT', p.IDENT)

    def build_from_mnemonic(self, mnemonic, operands):
        """
        Construye una representación interna de la instrucción a partir del mnemónico y operandos.
        
        Args:
            mnemonic (str): Nombre de la instrucción (ej: "ADD", "LW", "BEQ")
            operands (list): Lista de operandos parseados
        
        Returns:
            list: Representación interna de la instrucción con tipo, info y operandos
        
        Maneja:
        - Validación de operandos y rangos
        - Clasificación por tipo de instrucción (R, I, S, B, U, J)
        - Expansión de pseudoinstrucciones
        - Verificación de sintaxis específica por tipo
        """
        line_num = getattr(self, 'current_line', 0)
        
        # Manejar pseudoinstrucciones
        if mnemonic in PSEUDO_INSTRUCTIONS:
            args = []
            # Extraer y validar argumentos de la pseudoinstrucción
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

        # Validar registros en operandos básicos
        for i, op in enumerate(operands):
            if op[0] == 'REG' and not (0 <= op[1] <= 31):
                raise ValueError(f"Línea {line_num}: Registro x{op[1]} no válido (rango: x0-x31)")
            elif op[0] == 'MEM' and not (0 <= op[2] <= 31):
                raise ValueError(f"Línea {line_num}: Registro x{op[2]} no válido (rango: x0-x31)")

        # Normalizar operandos para facilitar procesamiento
        vals = []
        for op in operands:
            if op[0] == 'REG':
                vals.append(op[1])
            elif op[0] == 'NUMBER':
                vals.append(op[1])
            elif op[0] == 'IDENT':
                vals.append(op[1])
            elif op[0] == 'MEM':
                vals.append((op[1], op[2]))  # (offset, registro)
            else:
                vals.append(op)

        # ===== INSTRUCCIONES TIPO R =====
        # Formato: op rd, rs1, rs2 (registro-registro-registro)
        if mnemonic in ISA.get('R', {}):
            # Validar número de operandos
            if len(operands) != 3:
                raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' requiere 3 operandos, encontrados {len(operands)}")
            # Validar que todos sean registros
            for i, op in enumerate(operands):
                if op[0] != 'REG':
                    raise SyntaxError(f"Línea {line_num}: Operando {i+1} de '{mnemonic}' debe ser registro, encontrado {op[0]}")
            
            if len(vals) == 3 and all(isinstance(v, int) for v in vals):
                info = ISA['R'][mnemonic]
                return ["R", info, vals[0], vals[1], vals[2]]  # [tipo, info, rd, rs1, rs2]

        # ===== INSTRUCCIONES TIPO I =====
        # Formato: op rd, rs1, imm O op rd, offset(rs1) para loads
        if mnemonic in ISA.get('I', {}):
            # Casos especiales: EBREAK y ECALL no tienen operandos
            if mnemonic in ["EBREAK", "ECALL"]:
                if len(operands) != 0:
                    raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' no debe tener operandos")
                info = ISA['I'][mnemonic]
                # Usar el immediate definido en el JSON (0 para ECALL, 1 para EBREAK)
                immediate_value = int(info[2], 2)
                return ["I", info, 0, 0, immediate_value]
    
            # Validación especial para instrucciones de shift
            if mnemonic in ["SLLI", "SRLI", "SRAI"]:
                shift = vals[2]
                if not (0 <= shift < 32):   # RV32 soporta shifts de 0-31
                    raise ValueError(f"corrimiento inválido: {shift} en {mnemonic}")
    
            # Distinguir entre formato normal e instrucciones load
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
                    # Validar rango del offset para loads (-2048 a 2047)
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
                    # Validar rango del inmediato (-2048 a 2047)
                    if not (-2048 <= imm <= 2047):
                        raise ValueError(f"Línea {line_num}: Inmediato {imm} fuera de rango para tipo I (-2048 a 2047)")
                    info = ISA['I'][mnemonic]
                    return ["I", info, rd, rs1, imm]

        # ===== INSTRUCCIONES TIPO S =====
        # Formato: sw rs2, offset(rs1) (store)
        if mnemonic in ISA.get('S', {}):
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
                # Validar rango del offset para stores (-2048 a 2047)
                if not (-2048 <= offset <= 2047):
                    raise ValueError(f"Línea {line_num}: Offset {offset} fuera de rango para store (-2048 a 2047)")
                return ["S", info, rs2, rs1, offset]

        # ===== INSTRUCCIONES TIPO B =====
        # Formato: beq rs1, rs2, label (branch)
        if mnemonic in ISA.get('B', {}):
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
                return ["B", info, vals[0], vals[1], vals[2]]  # [tipo, info, rs1, rs2, label]

        # ===== INSTRUCCIONES TIPO U =====
        # Formato: lui rd, imm (upper immediate)
        if mnemonic in ISA.get('U', {}):
            if len(operands) != 2:
                raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' requiere 2 operandos, encontrados {len(operands)}")
            if operands[0][0] != 'REG':
                raise SyntaxError(f"Línea {line_num}: Primer operando de '{mnemonic}' debe ser registro")
            if operands[1][0] != 'NUMBER':
                raise SyntaxError(f"Línea {line_num}: Segundo operando de '{mnemonic}' debe ser inmediato")
            
            if len(vals) == 2:
                rd, imm = vals[0], vals[1]
                # Validar rango del inmediato (20 bits para tipo U)
                if not (-524288 <= imm <= 524287):
                    raise ValueError(f"Línea {line_num}: Inmediato {imm} fuera de rango para tipo U (-524288 a 524287)")
                info = ISA['U'][mnemonic]
                return ["U", info, rd, imm]

        # ===== INSTRUCCIONES TIPO J =====
        # Formato: jal rd, label (jump and link)
        if mnemonic in ISA.get('J', {}):
            if len(operands) != 2:
                raise SyntaxError(f"Línea {line_num}: Instrucción '{mnemonic}' requiere 2 operandos, encontrados {len(operands)}")
            if operands[0][0] != 'REG':
                raise SyntaxError(f"Línea {line_num}: Primer operando de '{mnemonic}' debe ser registro")
            if operands[1][0] != 'IDENT':
                raise SyntaxError(f"Línea {line_num}: Segundo operando de '{mnemonic}' debe ser etiqueta")
            
            if len(vals) == 2:
                info = ISA['J'][mnemonic]
                return ["J", info, vals[0], vals[1]]  # [tipo, info, rd, label]

        # Si llegamos aquí, la instrucción no se pudo interpretar
        raise SyntaxError(f"Línea {line_num}: No se pudo interpretar instrucción '{mnemonic}' con operandos {operands}")

# ========================
#  PRIMERA PASADA
# ========================
def first_pass(source_code):
    """
    Primera pasada del ensamblador: construir tabla de etiquetas y calcular direcciones.
    Ahora maneja secciones .text y .data, con directivas .word, .half, .byte.
    """
    labels = {}                    # Tabla de etiquetas
    instruction_addresses = {}     # Direcciones de instrucciones por línea
    data_addresses = {}            # Direcciones de datos por línea
    
    PC_text = 0x00000000           # Contador para instrucciones
    PC_data = 0x10000000           # Contador para datos (ejemplo base)
    section = ".text"              # Sección actual (default)
    
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
                if section == ".text":
                    labels[label] = PC_text
                elif section == ".data":
                    labels[label] = PC_data
            code = rest.strip()

        if not code:
            continue

        # Cambiar de sección
        if code.startswith(".text"):
            section = ".text"
            continue
        elif code.startswith(".data"):
            section = ".data"
            continue

        # Manejar directivas de datos
        if section == ".data":
            if code.startswith(".word"):
                valores = code.replace(".word", "").split(',')
                PC_data += 4 * len(valores)
                data_addresses[lineno] = PC_data
            elif code.startswith(".half"):
                valores = code.replace(".half", "").split(',')
                PC_data += 2 * len(valores)
                data_addresses[lineno] = PC_data
            elif code.startswith(".byte"):
                valores = code.replace(".byte", "").split(',')
                PC_data += 1 * len(valores)
                data_addresses[lineno] = PC_data
            continue

        # Manejar instrucciones
        if section == ".text":
            instruction_addresses[lineno] = PC_text
            PC_text += 4

    return labels, instruction_addresses, data_addresses, PC_text, PC_data

# ========================
#  SEGUNDA PASADA
# ========================
def assemble_instruction(instr, labels, instruction_addresses):
    """
    Convierte una instrucción parseada a código máquina de 32 bits.
    
    Args:
        instr (tuple): Instrucción parseada con formato [tipo, info, operandos...]
        labels (dict): Tabla de etiquetas para resolver saltos
        instruction_addresses (dict): Direcciones de instrucciones por línea
    
    Returns:
        int: Palabra de 32 bits con el código máquina, o None si hay error
    
    Esta función:
    1. Identifica el tipo de instrucción (R, I, S, B, U, J)
    2. Extrae los campos necesarios (opcode, funct3, funct7, etc.)
    3. Codifica los operandos en el formato binario correspondiente
    4. Resuelve las etiquetas para calcular offsets de saltos
    5. Ensambla la palabra final de 32 bits
    """
    instr_type = instr[0]
    
    # Solo procesar instrucciones válidas
    if instr_type not in ["R", "I", "S", "B", "U", "J"]:
        return None

    line_num = instr[-1]  # Número de línea (último elemento)
    pc = instruction_addresses.get(line_num)  # Dirección de esta instrucción
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
        
        # Verificar que el offset esté en rango y sea par (múltiplo de 2)
        if not (-4096 <= offset <= 4094) or offset % 2 != 0:
            raise ValueError(f"Línea {line_num}: Salto a '{label}' fuera de rango")

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
        
        # Verificar rango válido para JAL y que sea par
        if not (-1048576 <= offset <= 1048574) or offset % 2 != 0:
            raise ValueError(f"Línea {line_num}: Salto a '{label}' fuera de rango para JAL")

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
    Segunda pasada del ensamblador:
    - Convierte instrucciones a código máquina.
    - Convierte directivas de datos (.word, .half, .byte) a memoria de datos.
    
    Args:
        instructions (list): Lista de instrucciones y directivas parseadas.
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
            else:
                print(f"Advertencia: directiva {directive} no implementada")
            continue  # saltar a la siguiente

        # === MANEJO DE INSTRUCCIONES ===
        code_word = assemble_instruction(instr, labels, instruction_addresses)
        if code_word is not None:
            machine_code.append(code_word)

    return machine_code, data_memory


def expand_all_pseudo(instructions, lexer, parser):
    """
    Expandir todas las pseudoinstrucciones a instrucciones reales.
    
    Args:
        instructions (list): Lista de instrucciones parseadas (puede incluir pseudoinstrucciones)
        lexer: Analizador léxico para re-parsear instrucciones expandidas
        parser: Analizador sintáctico para re-parsear instrucciones expandidas
    
    Returns:
        list: Lista de instrucciones con todas las pseudoinstrucciones expandidas
    
    Las pseudoinstrucciones son instrucciones "sintéticas" que se traducen a una o más
    instrucciones reales. Por ejemplo:
    - LI x1, 100 se expande a ADDI x1, x0, 100
    - MV x1, x2 se expande a ADDI x1, x2, 0
    """
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
#  FUNCIÓN PRINCIPAL
# ========================
def main():
    """
    Función principal del ensamblador RISC-V RV32I.
    
    Proceso completo de ensamblado:
    1. Leer archivo de código assembly
    2. Primera pasada: construir tabla de etiquetas
    3. Análisis léxico y sintáctico
    4. Expansión de pseudoinstrucciones
    5. Segunda pasada: generar código máquina
    6. Escribir archivos de salida (hex y binario)
    7. Mostrar resultados en consola
    
    Archivos de salida:
    - output.hex: código máquina en formato hexadecimal
    - output.bin: código máquina en formato binario (texto)
    """
    print("=== Ensamblador RV32I ===")
    
    # ===== PASO 1: LEER ARCHIVO DE ENTRADA =====
    try:
        with open('ejemplo.asm', 'r', encoding='utf-8') as f:
            data = f.read()
        print("Archivo 'ejemplo.asm' leído correctamente")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'ejemplo.asm'")
        return
    except Exception as e:
        print(f"Error leyendo archivo: {e}")
        return

    # ===== PASO 2: INICIALIZAR ANALIZADORES =====
    lexer = RV32ILexer()    # Analizador léxico
    parser = AsmParser()    # Analizador sintáctico
    
    try:
        # ===== PASO 3: PRIMERA PASADA =====
        print("\n=== PRIMERA PASADA ===")
        labels, instruction_addresses, data_addresses, PC_text, PC_data = first_pass(data)
        print(f"Etiquetas encontradas: {labels}")
        print(f"Direcciones de instrucciones (texto): {instruction_addresses}")
        print(f"Direcciones de datos: {data_addresses}")
        print(f"PC final .text: 0x{PC_text:08X} ({PC_text} bytes)")
        print(f"PC final .data: 0x{PC_data:08X} ({PC_data} bytes)")


        
        # ===== PASO 4: ANÁLISIS SINTÁCTICO =====
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
        
        # ===== PASO 5: EXPANSIÓN DE PSEUDOINSTRUCCIONES =====
        print("Expandiendo pseudoinstrucciones...")
        final_instructions = expand_all_pseudo(parsed_instructions, lexer, parser)
        print(f"Total de instrucciones después de expansión: {len(final_instructions)}")
        
        # ===== PASO 6: SEGUNDA PASADA =====
        # SEGUNDA PASADA
        print("\n=== SEGUNDA PASADA ===")
        machine_code, data_memory = second_pass(final_instructions, labels, instruction_addresses, data_addresses)

        print(f"Código máquina generado: {len(machine_code)} instrucciones")
        print(f"Datos en memoria: {len(data_memory)} valores")

        
        if not machine_code:
            print("Error: No se generó código máquina")
            return
        
        print(f"Código máquina generado: {len(machine_code)} instrucciones")
        
        # ===== PASO 7: GENERAR ARCHIVOS DE SALIDA =====
        print("\n=== GENERANDO ARCHIVOS ===")

        with open("output_data.hex", "w") as f:
            for word in data_memory:
                f.write(f"{word:08x}\n")
        print("Archivo 'output_data.hex' generado")
        
        # Archivo hexadecimal (una palabra por línea)
        with open("output.hex", "w") as f:
            for word in machine_code:
                f.write(f"{word & 0xFFFFFFFF:08x}\n")
        print("Archivo 'output.hex' generado")
        
        # Archivo binario en formato texto (32 bits por línea)
        with open("output.bin", "w") as f:
            for word in machine_code:
                binary_str = f"{word & 0xFFFFFFFF:032b}"
                f.write(binary_str + "\n")
        print("Archivo 'output.bin' generado (formato texto binario)")
        
        # ===== PASO 8: MOSTRAR RESULTADOS =====
        print("\n=== CÓDIGO MÁQUINA ===")
        for i, word in enumerate(machine_code):
            pc_hex = f"{i*4:04x}"                    # Dirección en hex
            hex_word = f"{word & 0xFFFFFFFF:08x}"    # Palabra en hex
            bin_word = f"{word & 0xFFFFFFFF:032b}"   # Palabra en binario
            print(f"0x{pc_hex}: 0x{hex_word} | {bin_word}")
        
        print(f"\nEnsamblado completado exitosamente!")
        print(f"Total: {len(machine_code)} instrucciones ({len(machine_code)*4} bytes)")
        
    except Exception as e:
        print(f"Error durante el ensamblado: {e}")
        import traceback
        traceback.print_exc()

# ===== PUNTO DE ENTRADA =====
if __name__ == "__main__":
    main()