module pc(
  input wire clk,           // Botón físico
  input wire rst_n,         // Reset en bajo
  output reg [31:0] address, // Dirección actual
  output reg [31:0] next_pc
);

       // Declaramos next_pc

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      address  <= 32'b0;     // Reset: vuelve a 0
      next_pc  <= 32'b0;
    end else begin
      next_pc  <= address + 32'd4; // Calcula next_pc
      address  <= address + 32'd4; // Carga el nuevo valor en address
    end
  end
endmodule
