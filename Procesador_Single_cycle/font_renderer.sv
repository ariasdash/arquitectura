// ============================================================
// font_renderer.v 
// Renderizador de caracteres 8x16 usando font_rom
// ============================================================

module font_renderer (
    input  wire        clk,
    input  wire [7:0]  ascii_code,   // código ASCII
    input  wire [3:0]  row_in_char,  // 0–15
    input  wire [2:0]  col_in_char,  // 0–7
    output wire        pixel_on      // wire para lógica combinacional
);

    wire [10:0] rom_addr;
    wire [7:0]  rom_data;

    // Dirección dentro de la ROM: carácter * 16 + fila
    assign rom_addr = {ascii_code[6:0], row_in_char}; // Solo 7 bits de ASCII (0-127)

    font_rom rom_inst (
        .clk(clk),          
        .addr(rom_addr),
        .data(rom_data)
    );

    // Extrae el bit de la columna
    assign pixel_on = rom_data[7 - col_in_char];

endmodule