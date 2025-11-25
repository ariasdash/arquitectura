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
  011 -> tipo B (comparacion)
  100 -> tipo U
  101 -> tipo J
*/

always_comb begin
    case (imm_src)
        // Tipo I (ADDI, ANDI, ORI, etc.)
        3'b000: imm = {{20{instr[31]}}, instr[31:20]};
        
        // Tipo I de carga (LB, LH, LW, LBU, LHU)
        3'b001: imm = {{20{instr[31]}}, instr[31:20]};
        
        // Tipo S (SB, SH, SW)
        3'b010: imm = {{20{instr[31]}}, instr[31:25], instr[11:7]};
		  
		  // Tipo B
		  3'b011: imm = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
		  
		  //Tipo U
		  3'b100: imm = {instr[31:12], 12'b0};
		  
		  //Tipo J
		  3'b101: imm = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};

        
        default: imm = 32'b0;
    endcase
end
endmodule
