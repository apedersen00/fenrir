from vunit import VUnit

vu = VUnit.from_argv(compile_builtins=False)
vu.add_vhdl_builtins()

# Event capture tests
test_event_capture = vu.add_library("test_event_capture")
test_event_capture.add_source_files("conv_pool_pkg.vhd")
test_event_capture.add_source_files("event_capture.vhd")
test_event_capture.add_source_files("tb_event_capture.vhd")

# Convolution tests - ADD KERNEL WEIGHTS PACKAGE FIRST
test_convolution = vu.add_library("test_convolution")
test_convolution.add_source_files("conv_pool_pkg.vhd")
test_convolution.add_source_files("kernel_weights_pkg.vhd")      # ADD THIS LINE
test_convolution.add_source_files("convolution.vhd")
test_convolution.add_source_files("convolution_configurable.vhd")  # ADD THIS LINE
test_convolution.add_source_files("tb_convolution.vhd")

# Pooling tests
test_pooling = vu.add_library("test_pooling")
test_pooling.add_source_files("conv_pool_pkg.vhd")
test_pooling.add_source_files("pooling.vhd")
test_pooling.add_source_files("tb_pooling.vhd")

# Top module integration tests - ADD KERNEL WEIGHTS PACKAGE
test_top_module = vu.add_library("test_top_module")
test_top_module.add_source_files("conv_pool_pkg.vhd")
test_top_module.add_source_files("kernel_weights_pkg.vhd")       # ADD THIS LINE
test_top_module.add_source_files("event_capture.vhd")
test_top_module.add_source_files("convolution.vhd")
test_top_module.add_source_files("convolution_configurable.vhd")  # ADD THIS LINE
test_top_module.add_source_files("pooling.vhd")
test_top_module.add_source_files("dp_bram.vhd")
test_top_module.add_source_files("snn_processor_top.vhd")
test_top_module.add_source_files("tb_snn_processor_top.vhd")

# File-based verification tests - ADD KERNEL WEIGHTS PACKAGE
test_verification = vu.add_library("test_verification")
test_verification.add_source_files("conv_pool_pkg.vhd")
test_verification.add_source_files("kernel_weights_pkg.vhd")      # ADD THIS LINE
test_verification.add_source_files("event_capture.vhd")
test_verification.add_source_files("convolution.vhd")
test_verification.add_source_files("convolution_configurable.vhd")  # ADD THIS LINE
test_verification.add_source_files("pooling.vhd")
test_verification.add_source_files("dp_bram.vhd")
test_verification.add_source_files("snn_processor_top.vhd")
test_verification.add_source_files("tb_snn_verification.vhd")

# Add this line to enable VCD export
vu.set_sim_option("ghdl.sim_flags", ["--vcd=wave.vcd"])

vu.main()