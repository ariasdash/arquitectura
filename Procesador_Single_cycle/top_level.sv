module top_level (
	// CPU Control (Botones)
	input logic clk,             // Instruction step clock (Boton fisico/reloj lento)
	input logic rst_n,           // Active-low reset (Boton)
	input logic vgarstl,         // reset vga

	// VGA Clock Input (50MHz de la placa)
	input logic vga_clock_in,

	// LEDs & Displays
	input logic sw0,
	input logic sw1,
	input logic sw2,
	input logic sw3,
	output logic [6:0] display0,
	output logic [6:0] display1,
	output logic [6:0] display2,
	output logic [6:0] display3,
	output logic [9:0] leds,

	// VGA Outputs
	output logic [7:0] VGA_R,
	output logic [7:0] VGA_G,
	output logic [7:0] VGA_B,
	output logic VGA_HS,
	output logic VGA_VS,
	output logic VGA_CLK
);

	// =====================
	// 1. Reset Logic
	// =====================
	// Convierte el reset activo-bajo del boton (rst_n) a un reset activo-alto para la CPU y VGA.
	wire system_reset = ~rst_n;  // Señal de reset activa en alto
	wire vgarst = ~vgarstl;

	// =====================
	// 2. Señales internas (Wires para el Datapath)
	// =====================
	logic [31:0] pc_addr, next_pc;
	logic [31:0] instr;
	logic [3:0] aluOp;
	logic regWrite;
	logic [4:0] rs1, rs2, rd;
	logic [31:0] reg_data1, reg_data2;
	logic [31:0] alu_result;

	logic [31:0] imm;
	logic [31:0] alu_B, alu_A;                // salida del MUX (segundo operando de la ALU)
	logic [2:0] imm_src;
	logic [2:0] brOp;
	logic aluB_src;
	logic MemRead, MemWrite;
   logic [1:0]  MemToReg;
	logic  branch_taken;
	logic is_branch_instr;
	logic [31:0] branch_target;
	logic is_jalr_w;
	logic is_jal_w;
	logic [31:0] pc_4;
	logic halt_s;

	logic [31:0] mem_read_data;        // dato leido de memoria
	logic [31:0] write_back_data;      // dato final que vuelve al registro (ALU o Memoria)

	//  pasa de [31:0] a [7:0] para coincidir con el puerto 'mem' de color
	logic [7:0] mem_debug_wire [0:127]; // memoria de datos (array de bytes)

	logic [31:0] regs_debug [31:0]; // registros

	//  Wires para conectar color.sv <-> InstructionMemory
	logic [6:0]  imem_debug_addr_wire;
	logic [31:0] imem_debug_data_wire;
	
	logic pc_src;
   assign pc_src = (branch_taken & is_branch_instr) | is_jal_w;
	
	assign pc_4 = pc_addr + 4;
	
	
	//mux del pc
always_comb begin
        if (halt_s) begin
            // EBREAK: Congelar el PC (Next = Current)
            // El procesador se queda "atascado" aquí a propósito.
            next_pc = pc_addr; 
        end
        else if (is_jalr_w) begin
            // JALR (Prioridad 1)
            next_pc = alu_result & 32'hFFFFFFFE;
        end
        else if (pc_src) begin
            // Branch o JAL (Prioridad 2)
            next_pc = branch_target;
        end
        else begin
            // Flujo Normal (Prioridad 3)
            next_pc = pc_4;
        end
    end



	// =====================
	// 3. CPU Datapath Instantiations
	// =====================

	// PC (usa clk, el boton de avance)
	pc u_pc (
		.clk(clk),
		.rst_n(rst_n),
		.address(pc_addr),
		.next_pc(next_pc)
);
	// Instruction Memory
	InstructionMemory u_imem (
		.addr(pc_addr),
		.instr(instr),
		.debug_addr(imem_debug_addr_wire),
		.debug_data(imem_debug_data_wire)
	);

	// Decoder
	decoder u_dec (
		.instr(instr),
		.AluOp(aluOp),
		.regWrite(regWrite),
		.MemRead(MemRead),
		.MemWrite(MemWrite),
		.MemToReg(MemToReg),
		.rs1(rs1),
		.rs2(rs2),
		.rd(rd),
		.imm_src(imm_src),
		.aluB_src(aluB_src),
		.aluA_src(aluA_src),
		.brOp(brOp),
		.branch(is_branch_instr),
		.is_jalr(is_jalr_w),
		.is_jal(is_jal_w),
		.halt(halt_s)
	);

	// Register unit (usa clk, el boton de avance)
	register_unit u_regs (
		.clk(clk),
		.rst_n(rst_n),
		.RUWr(regWrite),
		.rs1(rs1),
		.rs2(rs2),
		.rd(rd),
		.data_in(write_back_data),
		.data_out1(reg_data1),
		.data_out2(reg_data2),
		.regs_debug(regs_debug)
	);

	// Generador de inmediatos
	imm_generator u_imm (
		.instr(instr),
		.imm_src(imm_src),
		.imm(imm)
	);

	// MUX para seleccionar segundo operando de la ALU
	mux2_1 B_mux (
		.x(reg_data2),
		.y(imm),
		.select(aluB_src),
		.r(alu_B)
	);

	// MUX para seleccionar primer operando de la ALU
	mux2_1 A_mux (
		.x(reg_data1),
		.y(pc_addr),
		.select(aluA_src),
		.r(alu_A)
	);
	
	// ALU
	Alu u_alu (
		.A(alu_A),
		.B(alu_B),
		.AluOp(aluOp),
		.AluResult(alu_result)
	);
	
	branch_unit u_branch_unit(
    .brOp(brOp),
    .rs1(reg_data1),
    .rs2(reg_data2),
    .pc(pc_addr),          // <-- PC actual
    .imm_branch(imm),  
    .branch_taken(branch_taken),
    .branch_target(branch_target)
);
	// DATA MEMORY (usa clk, el boton de avance)
	data_memory u_mem (
		.clk(clk),
		.MemRead(MemRead),
		.MemWrite(MemWrite),
		.funct3(instr[14:12]),
		.addr(alu_result),
		.write_data(reg_data2),
		.read_data(mem_read_data),
		.mem_debug(mem_debug_wire) // Pasa el array de bytes
	);

	// MUX de escritura al registro (MemToReg)
	mux_mem u_mux_mem (
		.alu_result(alu_result),
		.mem_data(mem_read_data),
		.MemToReg(MemToReg),
		.write_back(write_back_data),
		.pc_4(pc_4)
	);

	// =====================
	// 4. Salida en LEDs y Displays
	// =====================
	logic [31:0] debug_data;
	logic [2:0] selector;

	assign selector = {sw2, sw1, sw0};    // Combina los 3 switches en un numero

	// Define qué dato mostrar en los 7-segmentos y LEDs
	always_comb begin
		case (selector)
			3'b000: debug_data = instr;             // Muestra la Instrucción
			3'b001: debug_data = alu_result;        // Muestra el resultado de la ALU
			3'b010: debug_data = mem_read_data;     // Muestra el dato leído de Memoria
			3'b100: debug_data = write_back_data;   // Muestra el dato de Writeback (dato final escrito en registros)
			default: debug_data = 32'b0;
		endcase
	end

	assign leds = debug_data[9:0]; // 10 LSB en LEDs

	// Muestra el dato seleccionado en hexadecimal
	hex7seg h0 (.val(debug_data[3:0]),   .display(display0));
	hex7seg h1 (.val(debug_data[7:4]),   .display(display1));
	hex7seg h2 (.val(debug_data[11:8]),  .display(display2));
	hex7seg h3 (.val(debug_data[15:12]), .display(display3));

	// =====================
	// 5. VGA Color Module Instantiation (Debug Monitor)
	// =====================
	// Conecta las señales internas de la CPU al modulo de visualizacion VGA
	color u_color (
		.clock         (vga_clock_in),    // 50 MHz Input Clock
		.vgarst        (vgarst),          // System Reset (Activo Alto: ~vgarstl)

		// Debug Inputs from CPU
		.pc_addr           (pc_addr),          // Program Counter
		.instr             (instr),            // Instruction
		.alu_result        (alu_result),       // ALU Result
		.reg_data1         (reg_data1),        // Data read from RS1 
		.reg_data2         (reg_data2),        // Data read from RS2
		.rd_data           (rd),        // register destiny
		.regWrite          (regWrite),         // Control Signal: Register Write Enable
		.MemRead           (MemRead),          // Control Signal: Data Memory Read
		.MemWrite          (MemWrite),         // Control Signal: Data Memory Write
		.mem_read_data     (mem_read_data),    // Lectura de memoria
		.write_back_data   (write_back_data),  // Escritura de memoria
		.imm               (imm),              // Immediato
		.aluOp             (aluOp),
		.next_pc           (next_pc),
		.alu_B             (alu_B),
		.imm_src           (imm_src),
		.mem               (mem_debug_wire),   // Memoria de Datos (bytes)
		.regs_debug        (regs_debug),       // Registros
		.brOp					 (brOp),
		.branch_target	    (branch_target),
		.branch_taken		 (branch_taken),
		.is_jalr_w		    (is_jalr_w),
		.is_jal_w			 (is_jal_w),
		.MemToReg			 (MemToReg),
		.halt_s				 (halt_s),

		// [FIX 4: Puertos de depuración para Memoria de Instrucciones]
		.inst_mem_debug_addr(imem_debug_addr_wire),
		.inst_mem_debug_data(imem_debug_data_wire),

		// VGA Outputs
		.vga_red       (VGA_R),
		.vga_green     (VGA_G),
		.vga_blue      (VGA_B),
		.vga_hsync     (VGA_HS),
		.vga_vsync     (VGA_VS),
		.vga_clock     (VGA_CLK)
	);

endmodule