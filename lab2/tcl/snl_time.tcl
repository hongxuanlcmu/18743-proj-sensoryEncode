create_clock [get_ports clk]  -period {10000.00}  -waveform {0 5000} -name clk
set_clock_uncertainty 400 [get_clocks clk]
set_input_delay 2 -clock clk [remove_from_collection [all_inputs] clk]
set_output_delay 2 -clock clk [all_outputs]
set_load 1.5 [all_outputs]