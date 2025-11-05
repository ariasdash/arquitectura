module decoder( //control unit
    input  logic [31:0] instr,
    output logic [3:0]  AluOp,   
    output logic        regWrite,
    output logic [4:0]  rs1, rs2, rd,
	 output logic [2:0] imm_src,
	 output logic 		  aluB_src,
    output logic        MemRead,
    output logic        MemWrite,
    output logic        MemToReg
);

    logic [6:0] opcode;
    logic [2:0] funct3;
    logic [6:0] funct7;

    assign opcode = instr[6:0];
    assign rd     = instr[11:7];
    assign funct3 = instr[14:12];
    assign rs1    = instr[19:15];
    assign rs2    = instr[24:20];
    assign funct7 = instr[31:25];

    always_comb begin
    // Valores por defecto para evitar latches
    regWrite = 0;
    AluOp    = 4'b0000;
    imm_src  = 3'b000;
    aluB_src = 0;  // <-- ESTE es el importante
	MemRead  = 0;
    MemWrite = 0;
    MemToReg = 0;

    case (opcode)
        7'b0110011: begin  // Tipo R
            regWrite = 1;
            aluB_src = 0; // usa rs2
            case ({funct7, funct3})
                {7'b0000000, 3'b000}: AluOp = 4'b0000; // ADD
                {7'b0100000, 3'b000}: AluOp = 4'b0001; // SUB
                {7'b0000000, 3'b111}: AluOp = 4'b0011; // AND
                {7'b0000000, 3'b110}: AluOp = 4'b0100; // OR
                {7'b0000000, 3'b100}: AluOp = 4'b0010; // XOR
                {7'b0000000, 3'b001}: AluOp = 4'b0101; // SLL
                {7'b0000000, 3'b101}: AluOp = 4'b0110; // SRL
                {7'b0100000, 3'b101}: AluOp = 4'b0111; // SRA
                {7'b0000000, 3'b010}: AluOp = 4'b1000; // SLT
                {7'b0000000, 3'b011}: AluOp = 4'b1001; // SLTU
            endcase
        end

        7'b0010011: begin  // Tipo I aritméticas
            regWrite = 1;
            imm_src  = 3'b000;
            aluB_src = 1; // inmediato
            case(funct3)
                3'b000: AluOp = 4'b0000; // ADDI
                3'b100: AluOp = 4'b0010; // XORI
                3'b110: AluOp = 4'b0100; // ORI
                3'b111: AluOp = 4'b0011; // ANDI
                3'b001: AluOp = 4'b0101; // SLLI
                3'b101: begin
                    if (funct7 == 7'b0000000)
                        AluOp = 4'b0110; // SRLI
                    else if (funct7 == 7'b0100000)
                        AluOp = 4'b0111; // SRAI
                end
                3'b010: AluOp = 4'b1000; // SLTI
                3'b011: AluOp = 4'b1001; // SLTIU
            endcase
        end

        7'b0000011: begin  // Tipo I de carga
            regWrite = 1;
            imm_src  = 3'b001;
            aluB_src = 1; // inmediato
            AluOp    = 4'b0000;
				MemRead  = 1;
            MemWrite = 0;
            MemToReg = 1;
        end
		  
		  7'b0100011: begin // tipo S
                regWrite = 0;        // no escribe en registros
                MemWrite = 1;        // habilita escritura en memoria
                aluB_src = 1;        // usa inmediato
                imm_src  = 3'b010;   // tipo S
                AluOp    = 4'b0000;  // suma para calcular dirección
            end

        default: begin
        end
    endcase
end


endmodule
