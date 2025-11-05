module color(
  input clock,               // 50 MHz clock
  input vgarst,              // reset

  // Señales del procesador
  input [31:0] pc_addr,
  input [31:0] instr,
  input [31:0] alu_result,
  input [31:0] reg_data1,
  input [31:0] reg_data2,
  input [31:0] rd_data,
  input [31:0] regWrite,
  input [31:0] MemRead,
  input [31:0] MemWrite,
  input [31:0] mem_read_data,
  input [31:0] write_back_data,
  input [31:0] imm,
  input [31:0] aluOp,
  input [31:0] next_pc,
  input [31:0] alu_B,
  input [31:0] imm_src,
  input [7:0]  mem [0:127],         // Data Memory (array de bytes)
  input [31:0] regs_debug [31:0],   // Register File

  //Puertos para la Memoria de Instrucciones
  output logic [6:0]  inst_mem_debug_addr, // Puerto de SALIDA para pedir una dirección
  input  logic [31:0] inst_mem_debug_data, // Puerto de ENTRADA para recibir el dato

  output reg [7:0] vga_red,
  output reg [7:0] vga_green,
  output reg [7:0] vga_blue,
  output vga_hsync,
  output vga_vsync,
  output vga_clock
);

  // Señales VGA
  wire [10:0] x;
  wire [9:0]  y;
  wire videoOn;
  wire vgaclk;

  // PLL VGA
  clock1280x800 vgaclock(
    .clock50(clock),
    .reset(vgarst),
    .vgaclk(vgaclk)
  );

  assign vga_clock = vgaclk;

  // Controlador VGA 1280x800
  vga_controller_1280x800 ctrl(
    .clk(vgaclk),
    .reset(vgarst),
    .video_on(videoOn),
    .hsync(vga_hsync),
    .vsync(vga_vsync),
    .hcount(x),
    .vcount(y)
  );

  // Instancia de 'font_renderer'
  wire pixel_on;
  wire [7:0] current_ascii;
  wire [3:0] row_in_char;
  wire [2:0] col_in_char;

  font_renderer font_inst (
     .clk(vgaclk),
     .ascii_code(current_ascii),
     .row_in_char(row_in_char),
     .col_in_char(col_in_char),
     .pixel_on(pixel_on)
  );


  // --- Constantes de Layout ---
  localparam CHAR_W = 8;
  localparam CHAR_H = 16;
  localparam CHARS_PER_LINE = 20; // Solo usamos 20 caracteres de ancho
  localparam LINES_PER_COL = 33;  // Solo usamos 40 líneas de alto
  localparam LINE_SPACING = CHAR_H + 4; // Espaciado vertical

  localparam TEXT_START_X_COL0 = 240;
  localparam TEXT_START_X_COL1 = TEXT_START_X_COL0 + (CHARS_PER_LINE*CHAR_W) + 20;
  localparam TEXT_START_X_COL2 = TEXT_START_X_COL1 + (CHARS_PER_LINE*CHAR_W) + 20;
  localparam TEXT_START_X_COL3 = TEXT_START_X_COL2 + (CHARS_PER_LINE*CHAR_W) + 20;
  localparam TEXT_START_Y = 40;

  // --- Buffers de Texto ---
  reg [7:0] text [0:LINES_PER_COL-1][0:CHARS_PER_LINE-1];
  reg [7:0] text2[0:LINES_PER_COL-1][0:CHARS_PER_LINE-1];
  reg [7:0] text3[0:LINES_PER_COL-1][0:CHARS_PER_LINE-1];
  reg [7:0] text4[0:LINES_PER_COL-1][0:CHARS_PER_LINE-1];

  // Función para convertir nibble (4 bits) a caracter ASCII
  function [7:0] num_to_ascii;
    input [3:0] n;
    num_to_ascii = (n < 10) ? (8'd48 + n) : (8'd65 + n - 10);
  endfunction

  //Contador para leer la memoria de instrucciones secuencialmente
  reg [4:0] imem_display_addr = 0;

  // Pedimos la dirección que indica el contador.
  assign inst_mem_debug_addr = imem_display_addr;


  // --- Lógica de Actualización del HUD ---
  always @(posedge vgaclk) begin
    integer i;
    integer base;
    byte b0,b1,b2,b3;
    logic [31:0] imem_word_data; // Variable para guardar el dato leído

    // --- Columna 0: Señales del CPU ---

    // INSTR
    text[0][0]  = "I"; text[0][1]  = "N"; text[0][2]  = "S"; text[0][3]  = "T";
    text[0][4]  = "R"; text[0][5]  = ":"; text[0][6]  = " ";
    text[0][7]  = num_to_ascii(instr[31:28]);
    text[0][8]  = num_to_ascii(instr[27:24]);
    text[0][9]  = num_to_ascii(instr[23:20]);
    text[0][10] = num_to_ascii(instr[19:16]);
    text[0][11] = num_to_ascii(instr[15:12]);
    text[0][12] = num_to_ascii(instr[11:8]);
    text[0][13] = num_to_ascii(instr[7:4]);
    text[0][14] = num_to_ascii(instr[3:0]);

    // PC
    text[1][0]  = "P"; text[1][1]  = "C"; text[1][2]  = ":"; text[1][3]  = " ";
    text[1][4]  = num_to_ascii(pc_addr[31:28]);
    text[1][5]  = num_to_ascii(pc_addr[27:24]);
    text[1][6]  = num_to_ascii(pc_addr[23:20]);
    text[1][7]  = num_to_ascii(pc_addr[19:16]);
    text[1][8]  = num_to_ascii(pc_addr[15:12]);
    text[1][9]  = num_to_ascii(pc_addr[11:8]);
    text[1][10] = num_to_ascii(pc_addr[7:4]);
    text[1][11] = num_to_ascii(pc_addr[3:0]);

    // RS1
    text[2][0]  = "R"; text[2][1]  = "S"; text[2][2]  = "1"; text[2][3]  = ":"; text[2][4]  = " ";
    text[2][5]  = num_to_ascii(reg_data1[31:28]);
    text[2][6]  = num_to_ascii(reg_data1[27:24]);
    text[2][7]  = num_to_ascii(reg_data1[23:20]);
    text[2][8]  = num_to_ascii(reg_data1[19:16]);
    text[2][9]  = num_to_ascii(reg_data1[15:12]);
    text[2][10] = num_to_ascii(reg_data1[11:8]);
    text[2][11] = num_to_ascii(reg_data1[7:4]);
    text[2][12] = num_to_ascii(reg_data1[3:0]);

    // RS2
    text[3][0]  = "R"; text[3][1]  = "S"; text[3][2]  = "2"; text[3][3]  = ":"; text[3][4]  = " ";
    text[3][5]  = num_to_ascii(reg_data2[31:28]);
    text[3][6]  = num_to_ascii(reg_data2[27:24]);
    text[3][7]  = num_to_ascii(reg_data2[23:20]);
    text[3][8]  = num_to_ascii(reg_data2[19:16]);
    text[3][9]  = num_to_ascii(reg_data2[15:12]);
    text[3][10] = num_to_ascii(reg_data2[11:8]);
    text[3][11] = num_to_ascii(reg_data2[7:4]);
    text[3][12] = num_to_ascii(reg_data2[3:0]);

    // RD (register destination)
    text[4][0]  = "R"; text[4][1]  = "D"; text[4][2]  = ":"; text[4][3]  = " ";
    text[4][4]  = num_to_ascii(rd_data[31:28]);
    text[4][5]  = num_to_ascii(rd_data[27:24]);
    text[4][6]  = num_to_ascii(rd_data[23:20]);
    text[4][7]  = num_to_ascii(rd_data[19:16]);
    text[4][8]  = num_to_ascii(rd_data[15:12]);
    text[4][9]  = num_to_ascii(rd_data[11:8]);
    text[4][10] = num_to_ascii(rd_data[7:4]);
    text[4][11] = num_to_ascii(rd_data[3:0]);

    // ALU Result
    text[5][0]  = "A"; text[5][1]  = "L"; text[5][2]  = "U"; text[5][3]  = ":"; text[5][4]  = " ";
    text[5][5]  = num_to_ascii(alu_result[31:28]);
    text[5][6]  = num_to_ascii(alu_result[27:24]);
    text[5][7]  = num_to_ascii(alu_result[23:20]);
    text[5][8]  = num_to_ascii(alu_result[19:16]);
    text[5][9]  = num_to_ascii(alu_result[15:12]);
    text[5][10] = num_to_ascii(alu_result[11:8]);
    text[5][11] = num_to_ascii(alu_result[7:4]);
    text[5][12] = num_to_ascii(alu_result[3:0]);

    // REGWRITE
    text[6][0]  = "R"; text[6][1]  = "W"; text[6][2]  = ":"; text[6][3]  = " ";
    text[6][4]  = num_to_ascii(regWrite[31:28]);
    text[6][5]  = num_to_ascii(regWrite[27:24]);
    text[6][6]  = num_to_ascii(regWrite[23:20]);
    text[6][7]  = num_to_ascii(regWrite[19:16]);
    text[6][8]  = num_to_ascii(regWrite[15:12]);
    text[6][9]  = num_to_ascii(regWrite[11:8]);
    text[6][10] = num_to_ascii(regWrite[7:4]);
    text[6][11] = num_to_ascii(regWrite[3:0]);

    // MEMREAD
    text[7][0]  = "M"; text[7][1]  = "R"; text[7][2]  = ":"; text[7][3]  = " ";
    text[7][4]  = num_to_ascii(MemRead[31:28]);
    text[7][5]  = num_to_ascii(MemRead[27:24]);
    text[7][6]  = num_to_ascii(MemRead[23:20]);
    text[7][7]  = num_to_ascii(MemRead[19:16]);
    text[7][8]  = num_to_ascii(MemRead[15:12]);
    text[7][9]  = num_to_ascii(MemRead[11:8]);
    text[7][10] = num_to_ascii(MemRead[7:4]);
    text[7][11] = num_to_ascii(MemRead[3:0]);

    // MEMWRITE
    text[8][0]  = "M"; text[8][1]  = "W"; text[8][2]  = ":"; text[8][3]  = " ";
    text[8][4]  = num_to_ascii(MemWrite[31:28]);
    text[8][5]  = num_to_ascii(MemWrite[27:24]);
    text[8][6]  = num_to_ascii(MemWrite[23:20]);
    text[8][7]  = num_to_ascii(MemWrite[19:16]);
    text[8][8]  = num_to_ascii(MemWrite[15:12]);
    text[8][9]  = num_to_ascii(MemWrite[11:8]);
    text[8][10] = num_to_ascii(MemWrite[7:4]);
    text[8][11] = num_to_ascii(MemWrite[3:0]);

    // MEM READ DATA
    text[9][0]  = "M"; text[9][1]  = "R"; text[9][2]  = "D"; text[9][3]  = ":"; text[9][4] = " ";
    text[9][5]  = num_to_ascii(mem_read_data[31:28]);
    text[9][6]  = num_to_ascii(mem_read_data[27:24]);
    text[9][7]  = num_to_ascii(mem_read_data[23:20]);
    text[9][8]  = num_to_ascii(mem_read_data[19:16]);
    text[9][9]  = num_to_ascii(mem_read_data[15:12]);
    text[9][10] = num_to_ascii(mem_read_data[11:8]);
    text[9][11] = num_to_ascii(mem_read_data[7:4]);
    text[9][12] = num_to_ascii(mem_read_data[3:0]);

    // WRITE BACK
    text[10][0] = "W"; text[10][1] = "B"; text[10][2] = ":"; text[10][3] = " ";
    text[10][4] = num_to_ascii(write_back_data[31:28]);
    text[10][5] = num_to_ascii(write_back_data[27:24]);
    text[10][6] = num_to_ascii(write_back_data[23:20]);
    text[10][7] = num_to_ascii(write_back_data[19:16]);
    text[10][8] = num_to_ascii(write_back_data[15:12]);
    text[10][9] = num_to_ascii(write_back_data[11:8]);
    text[10][10] = num_to_ascii(write_back_data[7:4]);
    text[10][11] = num_to_ascii(write_back_data[3:0]);

    // aluOp
    text[11][0] = "A"; text[11][1] = "O"; text[11][2] = "P"; text[11][3] = " ";
    text[11][4] = num_to_ascii(aluOp[31:28]);
    text[11][5] = num_to_ascii(aluOp[27:24]);
    text[11][6] = num_to_ascii(aluOp[23:20]);
    text[11][7] = num_to_ascii(aluOp[19:16]);
    text[11][8] = num_to_ascii(aluOp[15:12]);
    text[11][9] = num_to_ascii(aluOp[11:8]);
    text[11][10] = num_to_ascii(aluOp[7:4]);
    text[11][11] = num_to_ascii(aluOp[3:0]);

    // alu_B
    text[12][0] = "A"; text[12][1] = "L"; text[12][2] = "B"; text[12][3] = " ";
    text[12][4] = num_to_ascii(alu_B[31:28]);
    text[12][5] = num_to_ascii(alu_B[27:24]);
    text[12][6] = num_to_ascii(alu_B[23:20]);
    text[12][7] = num_to_ascii(alu_B[19:16]);
    text[12][8] = num_to_ascii(alu_B[15:12]);
    text[12][9] = num_to_ascii(alu_B[11:8]);
    text[12][10] = num_to_ascii(alu_B[7:4]);
    text[12][11] = num_to_ascii(alu_B[3:0]);

    // Imm
    text[13][0] = "I"; text[13][1] = "M"; text[13][2] = "M"; text[13][3] = " ";
    text[13][4] = num_to_ascii(imm[31:28]);
    text[13][5] = num_to_ascii(imm[27:24]);
    text[13][6] = num_to_ascii(imm[23:20]);
    text[13][7] = num_to_ascii(imm[19:16]);
    text[13][8] = num_to_ascii(imm[15:12]);
    text[13][9] = num_to_ascii(imm[11:8]);
    text[13][10] = num_to_ascii(imm[7:4]);
    text[13][11] = num_to_ascii(imm[3:0]);

    // imm_src
    text[14][0] = "I"; text[14][1] = "S"; text[14][2] = "C"; text[14][3] = " ";
    text[14][4] = num_to_ascii(imm_src[31:28]);
    text[14][5] = num_to_ascii(imm_src[27:24]);
    text[14][6] = num_to_ascii(imm_src[23:20]);
    text[14][7] = num_to_ascii(imm_src[19:16]);
    text[14][8] = num_to_ascii(imm_src[15:12]);
    text[14][9] = num_to_ascii(imm_src[11:8]);
    text[14][10] = num_to_ascii(imm_src[7:4]);
    text[14][11] = num_to_ascii(imm_src[3:0]);


    // --- Columna 1: Memoria de Datos (text2) ---
	 
	 text2[0][0] = "D"; text2[0][1] = "M"; text2[0][2] = "E"; text2[0][3] = "M"; 

    for(i=0; i<16; i++) begin
      base = i*4;
      b0 = mem[base+0];
      b1 = mem[base+1];
      b2 = mem[base+2];
      b3 = mem[base+3];
      
      // Dirección
      text2[i+1][0] = num_to_ascii(base[7:4]);
      text2[i+1][1] = num_to_ascii(base[3:0]);
      text2[i+1][2] = ":";
      text2[i+1][3] = " ";
      
      // Datos (4 bytes)
      text2[i+1][4] = num_to_ascii(b3[7:4]); // B3 (MSB)
      text2[i+1][5] = num_to_ascii(b3[3:0]);
      text2[i+1][6] = num_to_ascii(b2[7:4]); // B2
      text2[i+1][7] = num_to_ascii(b2[3:0]);
      text2[i+1][8] = num_to_ascii(b1[7:4]); // B1
      text2[i+1][9] = num_to_ascii(b1[3:0]);
      text2[i+1][10] = num_to_ascii(b0[7:4]); // B0 (LSB)
      text2[i+1][11] = num_to_ascii(b0[3:0]);
    end

    // --- Columna 2: Registros (text3) ---

	 text3[0][0] = "R"; text3[0][1] = "E"; text3[0][2] = "G"; text3[0][3] = "I"; 
    for(i = 0; i < 32; i++) begin

      // Registro (x00, x01, ..., x31)
      text3[i+1][0] = "x";
      text3[i+1][1] = num_to_ascii(i[4:4]);
      text3[i+1][2] = num_to_ascii(i[3:0]);
      text3[i+1][3] = ":";
      text3[i+1][4] = " ";

      // Valor (32 bits)
      text3[i+1][5]  = num_to_ascii(regs_debug[i][31:28]);
      text3[i+1][6]  = num_to_ascii(regs_debug[i][27:24]);
      text3[i+1][7]  = num_to_ascii(regs_debug[i][23:20]);
      text3[i+1][8]  = num_to_ascii(regs_debug[i][19:16]);
      text3[i+1][9]  = "_"; // separador
      text3[i+1][10] = num_to_ascii(regs_debug[i][15:12]);
      text3[i+1][11] = num_to_ascii(regs_debug[i][11:8]);
      text3[i+1][12] = num_to_ascii(regs_debug[i][7:4]);
      text3[i+1][13] = num_to_ascii(regs_debug[i][3:0]);
    end
	 
	 text4[0][0] = "I"; text4[0][1] = "M"; text4[0][2] = "E"; text4[0][3] = "M"; 

    base = imem_display_addr * 4; 
    
    // Se guarda el dato que acaba de llegar
    imem_word_data = inst_mem_debug_data; 

    // Se escribe la dirección (ej: "0000000C") en la línea 'imem_display_addr'
    text4[imem_display_addr + 1][0] = "0";
    text4[imem_display_addr + 1][1] = "x";
    text4[imem_display_addr + 1][2] = num_to_ascii(base[7:4]); // Mostrando solo 8 bits de dirección
    text4[imem_display_addr + 1][3] = num_to_ascii(base[3:0]);
    text4[imem_display_addr + 1][4] = ":";
    text4[imem_display_addr + 1][5] = " ";

    // se escribe el dato (la instrucción en sí)
    text4[imem_display_addr + 1][6]  = num_to_ascii(imem_word_data[31:28]);
    text4[imem_display_addr + 1][7]  = num_to_ascii(imem_word_data[27:24]);
    text4[imem_display_addr + 1][8]  = num_to_ascii(imem_word_data[23:20]);
    text4[imem_display_addr + 1][9]  = num_to_ascii(imem_word_data[19:16]);
    text4[imem_display_addr + 1][10] = "_";
    text4[imem_display_addr + 1][11] = num_to_ascii(imem_word_data[15:12]);
    text4[imem_display_addr + 1][12] = num_to_ascii(imem_word_data[11:8]);
    text4[imem_display_addr + 1][13] = num_to_ascii(imem_word_data[7:4]);
    text4[imem_display_addr + 1][14] = num_to_ascii(imem_word_data[3:0]);

    // Se incrementa el contador para la proxima direccion a pedir
    if (imem_display_addr == 31) begin
      imem_display_addr <= 0; // Vuelve al inicio
    end else begin
      imem_display_addr <= imem_display_addr + 1; // Siguiente dirección
    end

  end 
  
  // Coordenadas relativas al texto
  wire [6:0] char_col_idx = 
      inside_text_col0 ? (x - TEXT_START_X_COL0) / CHAR_W :
      inside_text_col1 ? (x - TEXT_START_X_COL1) / CHAR_W :
      inside_text_col2 ? (x - TEXT_START_X_COL2) / CHAR_W :
      (x - TEXT_START_X_COL3) / CHAR_W;
      
  wire [5:0] line_num_idx = (y - TEXT_START_Y) / LINE_SPACING;

  // Verificamos si el pixel (x,y) está dentro de alguna columna de texto
  wire inside_text_col0 =
    (x >= TEXT_START_X_COL0) && (x < TEXT_START_X_COL0 + CHARS_PER_LINE*CHAR_W) &&
    (y >= TEXT_START_Y)      && (y < TEXT_START_Y + LINES_PER_COL*LINE_SPACING);

  wire inside_text_col1 =
    (x >= TEXT_START_X_COL1) && (x < TEXT_START_X_COL1 + CHARS_PER_LINE*CHAR_W) &&
    (y >= TEXT_START_Y)      && (y < TEXT_START_Y + LINES_PER_COL*LINE_SPACING);
    
  wire inside_text_col2 =
    (x >= TEXT_START_X_COL2) && (x < TEXT_START_X_COL2 + CHARS_PER_LINE*CHAR_W) &&
    (y >= TEXT_START_Y)      && (y < TEXT_START_Y + LINES_PER_COL*LINE_SPACING);
    
  wire inside_text_col3 =
    (x >= TEXT_START_X_COL3) && (x < TEXT_START_X_COL3 + CHARS_PER_LINE*CHAR_W) &&
    (y >= TEXT_START_Y)      && (y < TEXT_START_Y + LINES_PER_COL*LINE_SPACING);
    
  wire inside_any_text = inside_text_col0 || inside_text_col1 || inside_text_col2 || inside_text_col3;


  // Seleccionamos el caracter ASCII del buffer correspondiente
  assign current_ascii =
    inside_text_col0 ? text [line_num_idx][char_col_idx] :
    inside_text_col1 ? text2[line_num_idx][char_col_idx] :
    inside_text_col2 ? text3[line_num_idx][char_col_idx] :
    inside_text_col3 ? text4[line_num_idx][char_col_idx] :
    8'd32; // Espacio (default)

  // Coordenadas dentro del caracter (para el 'font_renderer')
  assign row_in_char = (y - TEXT_START_Y - line_num_idx*LINE_SPACING) % CHAR_H;
  
  assign col_in_char = 
      inside_text_col0 ? ( (x - TEXT_START_X_COL0) % CHAR_W ) :
      inside_text_col1 ? ( (x - TEXT_START_X_COL1) % CHAR_W ) :
      inside_text_col2 ? ( (x - TEXT_START_X_COL2) % CHAR_W ) :
      inside_text_col3 ? ( (x - TEXT_START_X_COL3) % CHAR_W ) :
      3'd0;

    wire [4:0] y_in_line = (y - TEXT_START_Y) % LINE_SPACING;
    wire is_in_text_gap = (y_in_line >= CHAR_H); // True si y_in_line es 16, 17, 18, o 19

  // --- Asignación Final de Color ---
  always @(*) begin
  

    if (~videoOn)
      {vga_red, vga_green, vga_blue} = 24'h000000;
      
    // Si estamos en el gap de 4 pixeles, forzar fondo
    else if (is_in_text_gap)
      {vga_red, vga_green, vga_blue} = 24'h001020;

    // Si la linea es valida, NO estamos en el gap, Y el pixel esta encendido, mostrar texto
    else if ((inside_text_col0 || inside_text_col1 || inside_text_col2 || inside_text_col3 ) && pixel_on)
      {vga_red, vga_green, vga_blue} = 24'hFFFFFF;
    
    // De lo contrario, fondo
    else
      {vga_red, vga_green, vga_blue} = 24'h001020;
  end

endmodule

// ============================================================
// Generador de reloj VGA
// ============================================================
module clock1280x800(clock50, reset, vgaclk);
  input clock50;
  input reset;
  output vgaclk;

  vgaClock clk(
    .ref_clk_clk(clock50),
    .ref_reset_reset(reset),
    .reset_source_reset(vgarst),
    .vga_clk_clk(vgaclk)
  );
endmodule


// ============================================================
// Controlador VGA 1280x800
// ============================================================
module vga_controller_1280x800 (
  input clk,
  input reset,
  output wire hsync,
  output wire vsync,
  output reg [10:0] hcount,
  output reg [9:0]  vcount,
  output video_on
);

  parameter H_VISIBLE = 1280;
  parameter H_FP      = 48;
  parameter H_SYNC    = 32;
  parameter H_BP      = 80;
  parameter H_TOTAL   = H_VISIBLE + H_FP + H_SYNC + H_BP;

  parameter V_VISIBLE = 800;
  parameter V_FP      = 3;
  parameter V_SYNC    = 6;
  parameter V_BP      = 22;
  parameter V_TOTAL   = V_VISIBLE + V_FP + V_SYNC + V_BP;

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      hcount <= 0;
      vcount <= 0;
    end else begin
      if (hcount == H_TOTAL - 1) begin
        hcount <= 0;
        if (vcount == V_TOTAL - 1)
          vcount <= 0;
        else
          vcount <= vcount + 1;
      end else begin
        hcount <= hcount + 1;
      end
    end
  end

  assign hsync = (hcount >= H_VISIBLE + H_FP) && 
                 (hcount < H_VISIBLE + H_FP + H_SYNC);
  assign vsync = (vcount >= V_VISIBLE + V_FP) && 
                 (vcount < V_VISIBLE + V_FP + V_SYNC);
  assign video_on = (hcount < H_VISIBLE) && (vcount < V_VISIBLE);
endmodule