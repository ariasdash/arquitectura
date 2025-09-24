# PROYECTO 1 : ENSAMBLADOR DE RISCV RV32I

## DESCRIPCIÓN GENERAL:
Este programa es un ensamblador completo para la arquitectura RISC-V RV32I que 
convierte código assembly legible a código máquina ejecutable por el procesador.

## CARACTERÍSTICAS PRINCIPALES:
- Soporte completo para la ISA RV32I
- Manejo de pseudoinstrucciones
- Secciones .text y .data
- Directivas .word, .half, .byte y .string
- Generación de mapa de memoria detallado
- Soporte para archivos solo con datos
- Validación exhaustiva de sintaxis y rangos

## ESTRUCTURA GENERAL

1. CARGA DE DEFINICIONES ISA (Líneas 30-50)
2. MANEJO DE PSEUDOINSTRUCCIONES (Líneas 52-140)
3. ANÁLISIS LÉXICO - LEXER (Líneas 160-240)
4. ANÁLISIS SINTÁCTICO - PARSER (Líneas 250-420)
5. PRIMERA PASADA (Líneas 630-720)
6. SEGUNDA PASADA (Líneas 730-920)
7. FUNCIÓN PRINCIPAL (Líneas 990-1200)

## MANEJO DE DICCIONARIOS 

El ensamblador utiliza varios archivos JSON para definir la ISA:

1. Rtype.json: Instrucciones tipo R (registro-registro-registro)
2. Itype.json: Instrucciones tipo I (inmediato y loads)
3. Stype.json: Instrucciones tipo S (stores)
4. Btype.json: Instrucciones tipo B (branches/saltos condicionales)
5. Utype.json: Instrucciones tipo U (upper immediate)
6. Jtype.json: Instrucciones tipo J (jumps/saltos incondicionales)
7. pseudo.json: Definiciones de pseudoinstrucciones
8. REGnames.json: Mapeo de nombres simbólicos de registros

FORMATO DE ARCHIVOS JSON:
Cada archivo contiene un diccionario donde:
- Clave: mnemónico de la instrucción
- Valor: lista con [opcode, funct3, funct7/immediate] en binario

## VALIDACION Y MANEJO DE ERRORES

El ensamblador realiza múltiples validaciones y manejo de errores:

1. REGISTROS:
   - Rango válido: x0-x31
   - Nombres simbólicos válidos

2. INMEDIATOS:
   - Tipo I: -2048 a 2047 (12 bits con signo)
   - Tipo S: -2048 a 2047 (12 bits con signo)
   - Tipo B: -4096 a 4094 (13 bits, múltiplo de 2)
   - Tipo U: -524288 a 524287 (20 bits con signo)
   - Tipo J: -1048576 a 1048574 (21 bits, múltiplo de 2)

3. SINTAXIS:
   - Número correcto de operandos por instrucción
   - Tipos correctos de operandos
   - Referencias válidas a etiquetas

4. SEMÁNTICA:
   - Etiquetas definidas antes de uso
   - Rangos de salto válidos
   - Alineación correcta para saltos

5. ERRORES LÉXICOS:
   - Caracteres no reconocidos
   - Formato inválido de números

6. ERRORES SINTÁCTICOS:
   - Sintaxis incorrecta de instrucciones
   - Número incorrecto de operandos
   - Tipos incorrectos de operandos

7. ERRORES SEMÁNTICOS:
   - Registros fuera de rango
   - Inmediatos fuera de rango
   - Etiquetas no definidas
   - Saltos fuera de rango

8. ERRORES DE ARCHIVO:
   - Archivo no encontrado
   - Problemas de codificación

Cada error incluye el número de línea donde ocurrió para facilitar la depuración.

## ARCHIVOS DE SALIDA 

1. output.hex:
   - Formato: Un valor hexadecimal de 8 dígitos por línea
   - Representa cada instrucción como palabra de 32 bits
   - Solo se genera si hay instrucciones en el programa
   - Útil para cargar en memoria o simuladores

2. output.bin:
   - Formato: Una cadena binaria de 32 dígitos por línea
   - Representa cada instrucción en binario puro
   - Solo se genera si hay instrucciones en el programa
   - Útil para análisis bit a bit

3. output_data.hex:
   - Formato: Un valor hexadecimal por línea
   - Representa cada dato (byte, half, word) o carácter de string
   - Solo se genera si hay directivas de datos
   - Útil para inicializar memoria de datos

4. memory_map.txt:
   - Formato: Tabla detallada con columnas: Etiqueta, Dirección, Tipo, Valor
   - Muestra la distribución completa de memoria de datos
   - Incluye información específica para cada tipo de dato:
     * .word, .half, .byte: valor en hex y decimal
     * .string: contenido del string y cantidad de bytes
   - Solo se genera si hay directivas de datos
   - Útil para debugging y verificación de layout de memoria

## EJEMPLOS DE USO 

CÓDIGO ASSEMBLY CON INSTRUCCIONES (ejemplo.asm):
```
.text
main:
    li x1, 10        # Cargar 10 en x1 (pseudoinstrucción)
    li x2, 20        # Cargar 20 en x2 (pseudoinstrucción)
    add x3, x1, x2   # x3 = x1 + x2 (instrucción real)
    beqz x3, end     # Si x3 es 0, saltar a end (pseudoinstrucción)
    addi x3, x3, 1   # x3 = x3 + 1 (instrucción real)
end:
    nop              # No operation (pseudoinstrucción)
```

CÓDIGO ASSEMBLY CON DATOS:
```
.data
num1:    .word 12345     # Palabra de 32 bits
message: .string "Hi"    # String con terminador null  
num2:    .half 1234      # Media palabra de 16 bits
flag:    .byte 1         # Byte de 8 bits
```

CÓDIGO ASSEMBLY MIXTO:
```
.data
counter: .word 0
msg:     .string "Counter: "

.text
main:
    lw x1, counter      # Cargar valor del contador
    addi x1, x1, 1      # Incrementar contador
    sw x1, counter      # Guardar nuevo valor
    beqz x1, end        # Si es cero, terminar
end:
    nop
```

PROCESO COMPLETO:
1. Primera pasada: 
   - .data: counter=0x10000000, msg=0x10000004
   - .text: main=0x00000000, end=0x00000010
2. Expansión: li → addi, beqz → beq, nop → addi x0,x0,0
3. Segunda pasada: Genera código máquina e inicializa datos
4. Salida: 
   - Instrucciones: 5 instrucciones, 20 bytes
   - Datos: 1 word + string "Counter: " (10 bytes total)

ARCHIVOS GENERADOS:
- output.hex: código máquina de instrucciones
- output_data.hex: valores de datos en hexadecimal
- memory_map.txt: mapa detallado de memoria

## MANEJO DE CASOS ESPECIALES 

1. ARCHIVOS SOLO CON DATOS:
   - El ensamblador puede procesar archivos que solo contengan sección .data
   - No requiere instrucciones para funcionar
   - Genera solo output_data.hex y memory_map.txt
   - Muestra advertencia: "Solo se generaron datos, sin instrucciones"

2. ARCHIVOS SOLO CON INSTRUCCIONES:
   - Procesa normalmente archivos sin sección .data
   - Genera solo output.hex y output.bin
   - Funciona como ensamblador tradicional

3. DETECCIÓN AUTOMÁTICA DE TIPOS:
   - El mapa de memoria detecta automáticamente el tipo de cada directiva
   - Muestra información específica según el tipo de dato
   - Para strings: muestra el contenido legible y cantidad de bytes

4. VALIDACIÓN DE SHIFT INSTRUCTIONS:
   - Las instrucciones de shift (SLLI, SRLI, SRAI) tienen validación especial
   - Verifican que el shift amount esté en rango 0-31
   - Usan codificación especial con funct7 + shamt
