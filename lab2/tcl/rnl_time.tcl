create_clock [get_ports aclk]  -period {10000.00}  -waveform {0 5000} -name aclk
create_clock [get_ports gclk]  -period {240000.00}  -waveform {0 120000} -name gclk

set_clock_uncertainty 400 [get_clocks aclk]
set_clock_uncertainty 9600 [get_clocks gclk]

set_input_delay 2 -clock aclk [remove_from_collection [all_inputs] aclk]
set_output_delay 2 -clock aclk [all_outputs]
set_load 1.5 [all_outputs]
