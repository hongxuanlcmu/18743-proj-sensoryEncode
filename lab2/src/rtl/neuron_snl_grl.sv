// Assumes 0->1 transitions as events or spikes
// Values are encoded as spiketimes
// Weights are thermometer-coded. For e.g., 11111000 represnts a weight of 3, 11000000 represents 6 and so on
// Models an edge temporal (GRL) neuron with step-no-leak response function

// Instructions -
// 1. Do not change the paths of the source codes.
// 2. Start bottom-top. Scroll to the bottom, start with bitonic_sort_2 module.
// 3. One possible approach - add modules for 4-input, 8-input and 16-input bitonic sorters (you can do so by copying the skeleton of 32-input bitonic sorter).
//    Then, complete the 32-bit implementation.
// 4. You can hierarchically verify your bitonic sorter running the 32-input bitonic_sort_tb file, either by adding debug statements in the tb file, or opening the 
//    simulation waveform window. Commands for running simulation is provided in the Makefile.

`timescale 1ns / 1ps


module top_wrapper (clk, clk_out, output_spike, input_spikes, input_weights);

    parameter SYN_SIZE = 4; // N - no. of input synapses
    parameter THRESHOLD = 11; // Theta - firing threshold of the neuron
    parameter RESP_FUN_PEAK = 8; // p - no. of up-steps/down-steps

    input [0:SYN_SIZE-1] input_spikes;
    input [0:SYN_SIZE-1][RESP_FUN_PEAK-1:0] input_weights;
    input clk;
    output output_spike;
    output clk_out;

    neuron_snl_grl nsg (.output_spike(output_spike), .input_spikes(input_spikes), .input_weights(input_weights));

    assign clk_out = clk;

endmodule


module neuron_snl_grl (output_spike, input_spikes, input_weights);

    parameter SYN_SIZE = 4; // N - no. of input synapses
    parameter THRESHOLD = 11; // Theta - firing threshold of the neuron
    parameter RESP_FUN_PEAK = 8; // p - no. of up-steps/down-steps
    localparam SORT_SIZE = SYN_SIZE*RESP_FUN_PEAK;
    genvar i, j, k;

    input [0:SYN_SIZE-1] input_spikes;
    input [0:SYN_SIZE-1][RESP_FUN_PEAK-1:0] input_weights;
    output output_spike;

    wire [0:SORT_SIZE-1] up_times;
    wire [0:SORT_SIZE-1] up_sort_out;
    
    // Up-step generation from input using weights
    generate

    for (i = 0; i < SYN_SIZE; i = i + 1)
    begin: loop1

        for (j = 0; j < RESP_FUN_PEAK; j = j + 1)
        begin: loop2
        
            or g1 (up_times[i*RESP_FUN_PEAK+j], input_spikes[i], input_weights[i][j]);

        end

    end

    endgenerate

    // Sorter
    bitonic_sort_32 up (.sorted_out(up_sort_out[0:SORT_SIZE-1]), .raw_in(up_times[0:SORT_SIZE-1]));

    assign output_spike = up_sort_out[THRESHOLD-1];

endmodule

module onoff_filter_comp_center (filter_center_in, filter_edge_in, on_center_index, off_center_index, on_center_out, off_center_out,);

    // (3,3) (5,5) (7,6) (9,7)
    parameter FILTER_WIDTH = 9;
    parameter SORTER_WIDTH = 7;
    localparam NUM_EDGE_PIXELS = FILTER_WIDTH**2 - 1;
    localparam NUM_SORTER_BITS = 2**SORTER_WIDTH;
    localparam NUM_PAD_BITS = NUM_SORTER_BITS - NUM_EDGE_PIXELS;
    localparam NUM_EDGE_PIXELS_HALF = NUM_EDGE_PIXELS / 2;
    // localparam ON_CENTER_INDEX = NUM_PAD_BITS + NUM_EDGE_PIXELS_HALF - 1;
    // localparam OFF_CENTER_INDEX = NUM_SORTER_BITS - NUM_EDGE_PIXELS_HALF;

    input filter_center_in;
    input [0:NUM_EDGE_PIXELS - 1] filter_edge_in;
    // Indices for sorted_edges
    input [SORTER_WIDTH-1:0] on_center_index;
    input [SORTER_WIDTH-1:0] off_center_index;
    output on_center_out;
    output off_center_out;

    logic [0:NUM_SORTER_BITS - 1] sorted_edges;

// 0 1 2 3 4 5 6 7
// 7 6 5 4 3 2 1 0

// 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
// in in in in in in in in 23 22 21 20 19 18 17 16 15 14 13 12 11 10 09 08 07 06 05 04 03 02 01 00

    bitonic_sort_32 #(.N(SORTER_WIDTH)) edgeSort8 (.sorted_out(sorted_edges), .raw_in({filter_edge_in, {NUM_PAD_BITS{1'b0}}}));

    // For on center, if filter_center_in earlier than sorted_edges[3], pass filter_center_in, else inhibit
    // Means earlier than half.
    // earlier_than isOnCenter (.a(filter_center_in), .b(sorted_edges[3]), .y(on_center_out));
    earlier_than isOnCenter (.a(filter_center_in), .b(sorted_edges[on_center_index]), .y(on_center_out));

    // For off center, if filter_center_in later than sorted_edges[4], pass sorted_edges[4], else inhibit
    // Means later than half.
    // earlier_than isOffCenter (.a(sorted_edges[4]), .b(filter_center_in), .y(off_center_out));
    earlier_than isOffCenter (.a(sorted_edges[off_center_index]), .b(filter_center_in), .y(off_center_out));

endmodule

// NOTE if a arrives before b, then y = a; otherwise y = 0;
// a arrives before b: posedge a while b == 0
// reset posedge at next gamma cycle
module earlier_than (
    input logic a,
    input logic b,
    output logic y
);
    // b must be reset to 0 for a cycle at next gamma cycle to reset latch
    always_latch begin
        if(!b) y <= a;
    end

endmodule


// NOTE A signal uses a rising edge to indicate a spike.
//      The bitonic_sort_32 sorts sorts the later spikes to the lower indices.
//      To sort 8 signals, use bitonic_sort_32 #(3, 8).
module bitonic_sort_32 (sorted_out, raw_in); 

    parameter N = 5;
    parameter INPUT_SIZE = 1<<N;
    genvar i, j, k;

    input [0:INPUT_SIZE-1] raw_in;
    output [0:INPUT_SIZE-1] sorted_out;

    /* Declare any intermediate wires you use */
    localparam HALF_SIZE = INPUT_SIZE/2;
    wire [0:HALF_SIZE-1] half_wires_0;
    wire [0:HALF_SIZE-1] half_wires_1;
    wire [0:INPUT_SIZE-1] inter_wires[N];



    /* Instantiate two 16-input sorters here  */
    generate
        if (N == 2) begin
            bitonic_sort_2 halfSorter1
                (half_wires_0, raw_in[0+:HALF_SIZE]);
            bitonic_sort_2 halfSorter2
                (half_wires_1, raw_in[(INPUT_SIZE-1)-:HALF_SIZE]);
        end
        else begin
            bitonic_sort_32 #(N-1, 1<<(N-1)) halfSorter1
                (half_wires_0, raw_in[0+:HALF_SIZE]);
            bitonic_sort_32 #(N-1, 1<<(N-1)) halfSorter2
                (half_wires_1, raw_in[(INPUT_SIZE-1)-:HALF_SIZE]);
        end
    endgenerate
    assign inter_wires[0][0+:HALF_SIZE] = half_wires_0;
    assign inter_wires[0][(INPUT_SIZE-1)-:HALF_SIZE] = {<<{half_wires_1}};



    /* WRITE YOUR CODE FOR THE LAST STAGE */

    // Hint: Use generate loops and instantiate 2-input bitonic sorter inside.
    // Syntax example as below:
    generate
        for (i = 0; i < N; i = i + 1) begin: loop1
            localparam STEP_SIZE = INPUT_SIZE >> (i + 1);
            for (j = 0; j < (1 << i); j = j + 1) begin: loop2
                for (k = 0; k < STEP_SIZE; k = k + 1) begin: loop3
                    if (i == N - 1)
                        bitonic_sort_2 last_step_sort
                            ({sorted_out[k+j*STEP_SIZE*2],
                              sorted_out[STEP_SIZE+k+j*STEP_SIZE*2]},
                             {inter_wires[i][k+j*STEP_SIZE*2],
                              inter_wires[i][STEP_SIZE+k+j*STEP_SIZE*2]});
                    else
                        bitonic_sort_2 last_step_sort
                            ({inter_wires[i+1][k+j*STEP_SIZE*2],
                              inter_wires[i+1][STEP_SIZE+k+j*STEP_SIZE*2]},
                             {inter_wires[i][k+j*STEP_SIZE*2],
                              inter_wires[i][STEP_SIZE+k+j*STEP_SIZE*2]});
                end
            end
        end
    endgenerate

    // Note that the output from the bottom 16-input sorter has to be sorted in descending order.
    // To enforce that, you can just reverse the indices when you take in input from the bottom 16-input sorter.



endmodule



module bitonic_sort_2 (sorted_out, raw_in); 

    input [0:1] raw_in;
    output [0:1] sorted_out;

    /* WRITE YOUR CODE HERE */
    assign sorted_out[0] = raw_in[0] & raw_in[1];
    assign sorted_out[1] = raw_in[0] | raw_in[1];

    /* YOUR CODE ENDS HERE */

endmodule
