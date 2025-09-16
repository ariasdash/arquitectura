from sly import Lexer, Parser
import json
import os

# ========================
#  LEXER
# ========================
class RV32ILexer(Lexer):
    tokens = { 'INSTR', 'REG', 'NUMBER', 'COMMA', 'LPAREN', 'RPAREN' }
    ignore = ' \t'

    # Tokens simples
    COMMA  = r','
    LPAREN = r'\('
    RPAREN = r'\)'

    # Registros (x0, x1, ..., x31)
    @_(r'x[0-9]|x[12][0-9]|x3[01]')
    def REG(self, t):
        t.value = int(t.value[1:])  # guardar solo el número
        return t

    # Instrucciones (ADD, SUB, LW, SW, BEQ, BNE, LUI, AUIPC, JAL, JALR, ADDI)
    @_(r'[A-Za-z][A-Za-z0-9]*')
    def INSTR(self, t):
        t.value = t.value.upper()  # Always convert to uppercase
        return t


    # Números inmediatos (decimales o hexadecimales)
    @_(r'-?(0x[0-9A-Fa-f]+|\d+)')
    def NUMBER(self, t):
        if t.value.startswith("0x"):
            t.value = int(t.value, 16)
        else:
            t.value = int(t.value)
        return t

    # Saltar nuevas líneas
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += len(t.value)

    def error(self, t):
        print(f"Caracter ilegal {t.value[0]!r}")
        self.index += 1


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


# ========================
#  PARSER
# ========================
class AsmParser(Parser):
    tokens = RV32ILexer.tokens

    # Start rule: a program is a list of instructions
    @_('instructions')
    def program(self, p):
        return p.instructions

    @_('instruction')
    def instructions(self, p):
        return [p.instruction]

    @_('instruction instructions')
    def instructions(self, p):
        return [p.instruction] + p.instructions

    # Connect all instruction types
    @_('instr_r')
    def instruction(self, p):
        return p.instr_r

    @_('instr_ib')
    def instruction(self, p):
        return p.instr_ib

    @_('instr_load_s')
    def instruction(self, p):
        return p.instr_load_s

    @_('instr_uj')
    def instruction(self, p):
        return p.instr_uj

    # --- R-type ---
    @_('INSTR REG COMMA REG COMMA REG')
    def instr_r(self, p):
        if p.INSTR not in ISA["R"]:
            raise SyntaxError(f"Instrucción R no reconocida: {p.INSTR}")
        info = ISA["R"][p.INSTR]
        return self.build_r(info, p.REG0, p.REG1, p.REG2)

    # --- I-type and B-type merged ---
    @_('INSTR REG COMMA REG COMMA NUMBER')
    def instr_ib(self, p):
        if p.INSTR in ISA["I"]:
            info = ISA["I"][p.INSTR]
            return self.build_i(info, p.REG0, p.REG1, p.NUMBER)
        elif p.INSTR in ISA["B"]:
            info = ISA["B"][p.INSTR]
            return self.build_b(info, p.REG0, p.REG1, p.NUMBER)
        else:
            raise SyntaxError(f"Instrucción I/B no reconocida: {p.INSTR}")

    # --- Load y S-type (con paréntesis) ---
    @_('INSTR REG COMMA NUMBER LPAREN REG RPAREN')
    def instr_load_s(self, p):
        if p.INSTR in ISA["I"]:
            info = ISA["I"][p.INSTR]
            return self.build_i(info, p.REG0, p.REG1, p.NUMBER)
        elif p.INSTR in ISA["S"]:
            info = ISA["S"][p.INSTR]
            return self.build_s(info, p.REG0, p.REG1, p.NUMBER)
        else:
            raise SyntaxError(f"Instrucción con paréntesis no reconocida: {p.INSTR}")

    # --- U-type and J-type merged ---
    @_('INSTR REG COMMA NUMBER')
    def instr_uj(self, p):
        if p.INSTR in ISA["U"]:
            info = ISA["U"][p.INSTR]
            return self.build_u(info, p.REG, p.NUMBER)
        elif p.INSTR in ISA["J"]:
            info = ISA["J"][p.INSTR]
            return self.build_j(info, p.REG, p.NUMBER)
        else:
            raise SyntaxError(f"Instrucción U/J no reconocida: {p.INSTR}")

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

# Example function to test the lexer and parser
def test_assembler():
    lexer = RV32ILexer()
    parser = AsmParser()
    # Example J-type instruction
    code = "JAL x1, 2048"
    tokens = list(lexer.tokenize(code))
    print("Tokens:", tokens)
    result = parser.parse(iter(tokens))
    print("Parse result:", result)

# Call the test function
if __name__ == "__main__":
    test_assembler()
