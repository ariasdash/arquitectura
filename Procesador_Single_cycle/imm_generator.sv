module imm_generator(
    input  logic [31:0] instr,     // instrucción completa
    input  logic [2:0]  imm_src,   // código del tipo de instrucción
    output logic [31:0] imm        // inmediato extendido a 32 bits
);

/*
  imm_src codes:
  000 -> tipo I aritmético-lógicas
  001 -> tipo I de carga
  010 -> tipo S (store: SB, SH, SW)
*/

always_comb begin
    case (imm_src)
        // Tipo I (ADDI, ANDI, ORI, etc.)
        3'b000: imm = {{20{instr[31]}}, instr[31:20]};
        
        // Tipo I de carga (LB, LH, LW, LBU, LHU)
        3'b001: imm = {{20{instr[31]}}, instr[31:20]};
        
        // Tipo S (SB, SH, SW)
        3'b010: imm = {{20{instr[31]}}, instr[31:25], instr[11:7]};
        
        default: imm = 32'b0;
    endcase
end
endmodule
