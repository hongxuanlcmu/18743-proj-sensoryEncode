/*
 * Instructions - Here, you are responsible for implementing the FSM for storing 3-bit synaptic weights (module is named synapse)
 *                For verifying your design, you can provide input stimuli to the synapse testbench skeleton provided to you  
 *                Commands for running the simulation are provided in the Makefile
 *










 * Implements an SRM0 neuron with ramp-no-leak (RNL) response function.
 * Uses temporal encoding - inputs are provided in terms of spikes and their values are encoded in the corresponding spiketimes.
 * STDP learning mechanism is not implemented as part of neuron here; it's implemented separately in the column.
 *
 * Assumptions : 1) weights are encoded as binary values.
 *               2) bit resolution for synaptic weights is 3, i.e., weights range from 0 to 7 (wmax).
 *               3) input spikes are encoded as pulses having a width of 8 (wmax+1) unit clock cycles.
 *               4) earliest input spike occurs atleast one unit clock period after the posedge of gamma clock, since STDP updates occur at posedge of gamma.
 *
 * Parameters  : INPUT_SIZE    - number of synapses per neuron (equivalent to 'P' in column)
 *               THRESHOLD     - spiking threshold for neuron
 *
 * Inputs      : input_spikes  - input spikes to the neuron encoded as 8-cycle wide pulses
 *               inc           - increment signals from STDP to increase corresponding synaptic weights
 *               dec           - decrement signals from STDP to decrease corresponding synaptic weights
 *               aclk          - unit clock for temporal encoding
 *               gclk          - gamma clock that separates computational waves
 *               grst          - 1-cycle wide pulse generated from gclk to reset intermediate signals between computational waves
 *               rst           - system reset (synchronous with gclk)
 * Outputs     : output_spike  - output spike of neuron encoded as an 8-cycle wide pulse
 *               weights       - the 3-bit synaptic weights of neuron to be used later in STDP
 */


module neuron_rnl_ptt (output_spike, weights, input_spikes, inc, dec, aclk, gclk, grst, rst);

    parameter INPUT_SIZE = 4;
    parameter THRESHOLD = 11;
    
    genvar i;

    input logic [0:INPUT_SIZE-1] input_spikes;
    input logic [0:INPUT_SIZE-1] inc;
    input logic [0:INPUT_SIZE-1] dec;
    input logic aclk, gclk, grst, rst;
    output logic output_spike;
    output logic [0:INPUT_SIZE-1][0:2] weights;

    logic [0:INPUT_SIZE-1] up_in;

    // Synaptic weight + readout logic FSM
    generate
    for (i = 0; i < INPUT_SIZE; i = i + 1)
    begin: fsm
    
        synapse f1 (.out(up_in[i]),
                        .weight(weights[i]),
                        .input_spike(input_spikes[i]),
                        .inc(inc[i]),
                        .dec(dec[i]),
                        .aclk(aclk),
                        .gclk(gclk),
                        .rst(rst)
                       );

    end
    endgenerate

    neuron_body #(INPUT_SIZE, THRESHOLD) p1 (.output_spike(output_spike),
                                             .acc_in(up_in),
                                             .aclk(aclk),
                                             .pac_rst(grst),
                                             .rst(rst)
                                            );


endmodule

/*
 *
 * Implements the body (soma) of an SRM0 neuron with ramp-no-leak (RNL) response function.
 * Accumulates response functions from all synapses and generates an output spike when threshold is crossed.
 *
 * Parameters  : INPUT_SIZE    - number of synapses per neuron (equivalent to 'P' in column)
 *               THRESHOLD     - spiking threshold for neuron
 *
 * Inputs      : acc_in        - unary outputs from synapses (1 bit output per synapse)
 *               aclk          - unit clock for temporal encoding
 *               pac_rst       - 1-cycle wide pulse generated from gclk to reset intermediate signals between computational waves
 *               rst           - system reset (synchronous with gclk)
 * Outputs     : output_spike  - output spike of neuron encoded as an 8-cycle wide pulse
 */


module neuron_body (output_spike, acc_in, aclk, pac_rst, rst);

    parameter INPUT_SIZE = 16;
    parameter THRESHOLD = 13;

    input logic [0:INPUT_SIZE-1] acc_in;
    input logic aclk, pac_rst, rst;
    output logic output_spike;

    logic temp_spike;

    pac #(INPUT_SIZE, THRESHOLD) p1 (.out(temp_spike),
                                     .in(acc_in),
                                     .aclk(aclk),
                                     .grst(pac_rst)
                                    );
 
    fsm_simple fs (.out(output_spike),
                   .in(temp_spike),
                   .aclk(aclk),
                   .rst(rst)
                  );

endmodule


/*
 *
 * Implements a parallel accumulative counter as part of the neuron body (soma).
 * Accumulates response functions from all synapses into the body potential.
 * Threshold comparison is also integrated into it.
 *
 * Parameters  : INPUT_SIZE    - number of synapses per neuron (equivalent to 'P' in column)
 *               THRESHOLD     - spiking threshold for neuron
 *
 * Inputs      : in            - unary outputs from synapses (1 bit output per synapse)
 *               aclk          - unit clock for temporal encoding
 *               grst          - 1-cycle wide pulse generated from gclk to reset intermediate signals between computational waves
 * Outputs     : out           - 1-cycle wide output pulse that is generated when body potential crosses threshold
 */

`define max(v1, v2) ((v1) > (v2) ? (v1) : (v2))

module pac (out, in, aclk, grst);

    parameter INPUT_SIZE = 32;
    parameter THRESHOLD = 13;
    
    localparam OUT_RES = $clog2(INPUT_SIZE);
    localparam IN_SIZE = (1<<OUT_RES);
    localparam STAGES = $clog2(IN_SIZE)-1;
    localparam NUM = 2*IN_SIZE - OUT_RES-2;
    localparam MAXRES = `max(OUT_RES+1,$clog2(THRESHOLD)+1);

    genvar i, j;

    input logic [0:INPUT_SIZE-1] in;
    input logic aclk, grst;
    output logic out;
   
    logic [0:IN_SIZE-1] tin;
    wire logic [0:NUM-1] temp;
    wire logic [0:OUT_RES-1] tout;
    wire logic [0:MAXRES] t2out;
    logic [0:MAXRES-1] fout;
    logic [0:MAXRES-1] muxout;

    assign tin = IN_SIZE'(in);

    for (i = 0; i < IN_SIZE/2; i = i + 1)
    begin

        assign temp[i] = tin[i];

    end

    generate
        
        for (i = 0; i < STAGES; i = i + 1)
        begin: stages

            for (j = 0; j < IN_SIZE/(1<<(i+2)); j = j + 1)
            begin: adders
                
                adder #(i+1) a1 (.out(temp[(IN_SIZE>>(i+1))*((1<<(i+2))-i-3) + j*(i+2) : (IN_SIZE>>(i+1))*((1<<(i+2))-i-3) + j*(i+2) + (i+1)]),
                                 .a(temp[(IN_SIZE>>i)*((1<<(i+1))-i-2) + 2*j*(i+1) : (IN_SIZE>>i)*((1<<(i+1))-i-2) + 2*j*(i+1) + i]),
                                 .b(temp[(IN_SIZE>>i)*((1<<(i+1))-i-2) + (2*j+1)*(i+1) : (IN_SIZE>>i)*((1<<(i+1))-i-2) + (2*j+1)*(i+1) + i]),
                                 .cin(tin[IN_SIZE/2 + (IN_SIZE>>(i+1))*((1<<i)-1) + j])
                                );
            
            end

        end

    endgenerate

    assign tout = temp[NUM-OUT_RES:NUM-1];

    adder #(MAXRES) b1 (.out(t2out),
                        .a(MAXRES'(tout)),
                        .b(fout),
                        .cin(tin[IN_SIZE-1])
                       );

    always_ff @(posedge aclk)
    begin
        
        fout <= muxout;

    end

    assign out = ~t2out[1];

    assign muxout = (out | grst) ? -1*THRESHOLD : t2out[1:MAXRES];

endmodule


/*
 *
 * Implements the FSM responsible for storing synaptic weight and reading it out into a unary output corresponding to RNL response function.
 * Counts down for the duration of the input pulse, eventually wrapping around resetting the original weights.
 * Output is high only until the FSM counts down to 0; it becomes low after that.
 * STDP increment/decrement occur at the onset of gamma clock.
 *
 * Assumptions : 1) weights are encoded as binary values.
 *               2) bit resolution for synaptic weights is 3, i.e., weights range from 0 to 7 (wmax).
 *               3) input spikes are encoded as pulses having a width of 8 (wmax+1) unit clock cycles.
 *               4) earliest input spike occurs atleast one unit clock period after the posedge of gamma clock, since STDP updates occur at posedge of gamma.
 *
 * Inputs      : input_spike   - input spikes to the synapse encoded as 8-cycle wide pulses
 *               inc           - increment signal from STDP to increase the synaptic weight
 *               dec           - decrement signal from STDP to decrease the synaptic weight
 *               aclk          - unit clock for temporal encoding
 *               gclk          - gamma clock that separates computational waves
 *               rst           - system reset (synchronous with gclk)
 * Outputs     : out           - unary RNL output
 *               weight        - the corresponding 3-bit synaptic weight
 */

module synapse (out, weight, input_spike, inc, dec, aclk, gclk, rst);

    input logic aclk, gclk, rst;
    input logic input_spike, inc, dec;
    output logic out;
    output logic [0:2] weight;

    logic out_latch;
    logic tclk, tinc, tdec, is_weight_7, is_weight_0;
    logic [2:0] case_bits;
    logic [0:2] weight_next;

    assign out = ~out_latch & input_spike;

    assign tclk = aclk & input_spike;
    assign tinc = ~input_spike & inc;
    assign tdec = ~input_spike & dec;
    assign is_weight_0 = ~(weight[0] | weight[1] | weight[2]);
    assign is_weight_7 = weight[0] & weight[1] & weight[2];

    assign case_bits = {tinc, tdec, rst};

    always_comb begin
        casex (case_bits)
            3'bxx1: weight_next = '0;
            3'b100: begin
                if (~is_weight_7)
                    weight_next = weight + 3'b1;
                else
                    weight_next = weight;
            end
            3'b010: begin
                if (~is_weight_0)
                    weight_next = weight - 3'b1;
                else
                    weight_next = weight;
            end
            default: weight_next = weight;
        endcase
    end

    always_ff @(posedge gclk, posedge is_weight_7) begin
        if (gclk) begin
            if (rst) begin
                out_latch <= 1'b0;
            end
            out_latch <= input_spike & weight[0] & weight[1] & weight[2];
        end
        else begin
            out_latch <= input_spike & weight[0] & weight[1] & weight[2];
        end
    end

    always_ff @(posedge tclk, posedge gclk) begin
        if (tclk) begin
            weight <= weight - 3'b1;
        end
        else begin
            weight <= weight_next;
        end
    end

endmodule


/*
 *
 * Implements the FSM used to generate 8-cycles wide output spike pulses.
 *
 * Assumptions : 1) input is a pulse with a width of 1 unit clock period.
 *
 * Inputs      : in            - 1-cycle wide pulse to be converted to 8-cycles wide
 *               aclk          - unit clock for temporal encoding
 *               rst           - system reset (synchronous with gclk)
 * Outputs     : out           - 8-cycles wide pulse output
 */

module fsm_simple (out, in, aclk, rst);

    input logic aclk, rst;
    input logic in;
    output logic out;

    typedef enum logic [2:0] {S0, S1, S2, S3, S4, S5, S6, S7} state_t;
    state_t state;

    logic temp;

    always_ff @ (posedge aclk)
    begin

        if (rst)
        begin

            state <= S0;
        
        end
        else
        begin

            case(state)
            
                S0:
                begin
                    if (out) 
                    begin
                        state <= S1;
                    end
                    else
                    begin
                        state <= S0;
                    end
                end

                S1:
                begin
                    state <= S2;
                end

                S2:
                begin
                    state <= S3;
                end

                S3:
                begin
                    state <= S4;
                end
            
                S4:
                begin
                    state <= S5;
                end
            
                S5:
                begin
                    state <= S6;
                end

                S6:
                begin
                    state <= S7;
                end

                S7:
                begin
                    state <= S0;
                end
            
            endcase

        end

    end

    assign temp = ~(state[2] | state[1] | state[0]);
    assign out = (~temp) | (temp & in);

endmodule



/*
 *
 * Implements a simple multi-bit 2-input adder.
 *
 * Parameters : RESOLUTION - bit width of inputs
 *
 * Inputs     : a          - first data input
 *              b          - second data input
 *              cin        - 1-bit carry input
 * Outputs    : out        - sum output
 */



module adder (out, a, b, cin);

    parameter RESOLUTION = 4;

    input logic [0:RESOLUTION-1] a, b;
    input logic cin;
    output logic [0:RESOLUTION] out;

    assign out = a + b + cin;

endmodule
