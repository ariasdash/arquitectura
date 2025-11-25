module decoder( //control unit
    input  logic [31:0] instr,
    output logic [3:0]  AluOp,   
    output logic        regWrite,
    output logic [4:0]  rs1, rs2, rd,
	 output logic [2:0] imm_src,
	 output logic 		   aluB_src, aluA_src,
	 output logic [2:0]	brOp,
	 output logic 			branch, 
    output logic        MemRead,
    output logic        MemWrite,
    output logic [1:0]  MemToReg,
	 output logic 			is_jalr,
	 output logic 			is_jal,
	 output logic 			halt
	 
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
    regWrite = 0;
    AluOp    = 4'b0000;
    imm_src  = 3'b000;
    aluB_src = 0;
	 aluA_src = 0;
	 brOp 	 = 3'b010;
	 branch   = 0;
	 MemRead  = 0;
    MemWrite = 0;
    MemToReg = 2'b00;
	 is_jalr  = 0;
	 is_jal   =0;
	 halt		 =0;

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
		 7'b1100011: begin //tipo B
					branch = 1;
					aluA_src = 1; // usa pc como operando A
					aluB_src = 1;
					imm_src = 3'b011;
					AluOp   = 4'b0000;
					case(funct3)
						 3'b000: brOp = 3'b000; // BEQ
						 3'b001: brOp = 3'b001; // BNE
						 3'b100: brOp = 3'b100; // BLT
						 3'b101: brOp = 3'b101; // BGE
						 3'b110: brOp = 3'b110; // BLTU
						 3'b111: brOp = 3'b111; // BGEU
					 endcase
				end
				
			//tipo u 
			//lui
			7'b0110111: begin
				 regWrite = 1;
				 aluA_src = 0;
				 aluB_src = 1;
				 imm_src  = 3'b100;
				 AluOp    = 4'b1111; 
			end
			
			//auipc
			7'b0010111: begin 
				 regWrite = 1;
				 aluA_src = 1; 
				 aluB_src = 1;
				 imm_src  = 3'b100;
				 AluOp    = 4'b0000; 
			end
			
			// jalr
			7'b1100111: begin
            regWrite = 1;      
            imm_src  = 3'b000; 
            aluA_src = 0;      
            aluB_src = 1;      
            AluOp    = 4'b0000;
            MemToReg = 2'b10;  
            is_jalr  = 1;    
			end	
			//jal
			7'b1101111: begin 
				regWrite = 1;
            imm_src  = 3'b101; 
            aluA_src = 0;      
            aluB_src = 0;     
            AluOp    = 4'b0000; 
            MemToReg = 2'b10; 
            is_jal   = 1; 
			end
			//ebreak
			7'b1110011: begin
            if (funct3 == 3'b000) begin
               if (instr[20] == 1'b1) begin
                  halt = 1; 
               end
            end
         end
			


        default: begin
        end
    endcase
end


endmodule
