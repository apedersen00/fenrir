from vunit import VUnit

vu = VUnit.from_argv()
lib = vu.add_library("lib")
lib.add_source_files("package.conv_pool.vhd")
lib.add_source_files("dp_bram.vhd")
lib.add_source_files("module.conv_pool.vhd")
lib.add_source_files("tb_module.conv_pool.vhd")

# Add this line to enable VCD export
vu.set_sim_option("ghdl.sim_flags", ["--vcd=wave.vcd"])
#vu.set_compile_option("ghdl.a_flags", ["--std=08"])

vu.main()
