from vunit import VUnit

vu = VUnit.from_argv()
test_event_capture = vu.add_library("test_event_capture")
test_event_capture.add_source_files("conv_pool_pkg.vhd")
test_event_capture.add_source_files("event_capture.vhd")
test_event_capture.add_source_files("tb_event_capture.vhd")

# Add this line to enable VCD export
vu.set_sim_option("ghdl.sim_flags", ["--vcd=wave.vcd"])
#vu.set_compile_option("ghdl.a_flags", ["--std=08"])

vu.main()