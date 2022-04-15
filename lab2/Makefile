# Makefile for 18-743 Lab Assignment 2
# Author: Prabhu Vellaisamy


.PHONY: help
help:
	@echo ""
	@echo "	S Y N T A X "
	@echo ""
	@echo " make [args.] "
	@echo ""
	@echo " Arguments: "
	@echo "   bitonic-sim:  RTL sim. of bitonic sorter (32-bits) "
	@echo "   bitonic-wave:  waveform gui of bitonic sorter (32-bits) "
	@echo "   synapse-sim: RTL sim. of SNL neuron "
	@echo "   synapse-wave: waveform gui of SNL neuron "
	@echo "   pulse_gen-sim: RTL sim. of RNL neuron "
	@echo "   pulse_gen-wave: waveform gui of RNL neuron "
	@echo "   snl-synth: synthesis run for SNL neuron "
	@echo "   rnl-synth: synthesis run for RNL neuron "
	@echo ""
	@echo " "

SIM := ${CURDIR}/SIM
SRC:= ${CURDIR}/src
SYNTH := $(CURDIR)/SYNTH
BACK := /afs/andrew/course/18/743/backend

.SILENT:
.PHONY: bitonic-sim
bitonic-sim:
	
	if [ ! -d "$(SIM)" ]; then \
        mkdir $(SIM) && mkdir $(SIM)/bitonic_sort;\
		cd $(SIM)/bitonic_sort && vcs -sverilog -debug_all -full64 -top bitonic_sort_test $(SRC)/tb/bitonic_sort_test.sv $(SRC)/rtl/neuron_snl_grl.sv && ./simv;\
	elif [ ! -d "$(SIM)/bitonic_sort" ]; then \
        mkdir $(SIM)/bitonic_sort;\
		cd $(SIM)/bitonic_sort && vcs -sverilog -debug_all -full64 -top bitonic_sort_test $(SRC)/tb/bitonic_sort_test.sv $(SRC)/rtl/neuron_snl_grl.sv && ./simv;\
	else \
		cd $(SIM)/bitonic_sort && vcs -sverilog -debug_all -full64 -top bitonic_sort_test $(SRC)/tb/bitonic_sort_test.sv $(SRC)/rtl/neuron_snl_grl.sv && ./simv;\
	fi;
	

.PHONY: bitonic-wave
bitonic-wave:
	
	if [ ! -d "$(SIM)" ]; then \
        @echo "Perform simulation first";\
	elif [ ! -d "$(SIM)/bitonic_sort" ]; then \
        @echo "Perform simulation first";\
	else \
		cd $(SIM)/bitonic_sort && ./simv -gui;\
	fi;
	
	

.PHONY: synapse-sim
synapse-sim:

	if [ ! -d "$(SIM)" ]; then \
        mkdir $(SIM) && mkdir $(SIM)/synapse;\
		cd $(SIM)/synapse && vcs -sverilog -debug_all -full64 -top synapse $(SRC)/tb/synapse_tb.sv $(SRC)/rtl/neuron_rnl_ptt.sv && ./simv ; \
	elif [ ! -d "$(SIM)/synapse" ]; then \
        mkdir $(SIM)/synapse;\
		cd $(SIM)/synapse && vcs -sverilog -debug_all -full64 -top synapse $(SRC)/tb/synapse_tb.sv $(SRC)/rtl/neuron_rnl_ptt.sv && ./simv ; \
	else \
		cd $(SIM)/synapse && vcs -sverilog -debug_all -full64 -top synapse $(SRC)/tb/synapse_tb.sv $(SRC)/rtl/neuron_rnl_ptt.sv && ./simv ; \
	fi; 



.PHONY: synapse-wave
synapse-wave:
	if [ ! -d "$(SIM)" ]; then \
        @echo "Perform simulation first";\
	elif [ ! -d "$(SIM)/synapse" ]; then \
        @echo "Perform simulation first";\
	else \
		cd $(SIM)/synapse && ./simv -gui;\
	fi;


.PHONY: snl-synth
snl-synth:
	
	if [ ! -d "$(SYNTH)" ]; then \
		mkdir $(SYNTH) && mkdir $(SYNTH)/snl_neuron;\
		cd $(SYNTH)/snl_neuron && dc_shell -f $(CURDIR)/tcl/synth_snl.tcl;\
	elif [ ! -d "$(SYNTH)/snl_neuron" ]; then \
		mkdir $(SYNTH)/snl_neuron;\
		cd $(SYNTH)/snl_neuron && dc_shell -f $(CURDIR)/tcl/synth_snl.tcl;\
	else \
		cd $(SYNTH)/snl_neuron && dc_shell -f $(CURDIR)/tcl/synth_snl.tcl;\
	fi;
	
.PHONY: rnl-synth
rnl-synth:
		
	if [ ! -d "$(SYNTH)" ]; then \
		mkdir $(SYNTH) && mkdir $(SYNTH)/rnl_neuron;\
		cd $(SYNTH)/rnl_neuron && dc_shell -f $(CURDIR)/tcl/synth_rnl.tcl;\
	elif [ ! -d "$(SYNTH)/rnl_neuron" ]; then \
		mkdir $(SYNTH)/rnl_neuron;\
		cd $(SYNTH)/rnl_neuron && dc_shell -f $(CURDIR)/tcl/synth_rnl.tcl;\
	else \
		cd $(SYNTH)/rnl_neuron && dc_shell -f $(CURDIR)/tcl/synth_rnl.tcl;\
	fi;
	

.PHONY: cleanall
cleanall:
	rm -rf $(CURDIR)/SIM $(CURDIR)/SYNTH $(CURDIR)/PNR
	#find . -maxdepth 1 -type d | grep "./" | xargs -I {} make clean -C {}
