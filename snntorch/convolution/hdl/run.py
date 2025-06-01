from vunit import VUnit

vu = VUnit.from_argv()
lib_mccp = vu.add_library("multi_channel_conv_pool")
lib_mccp.add_source_files("package.conv_pool.vhd")
lib_mccp.add_source_files("dp_bram.vhd")
lib_mccp.add_source_files("module.conv_pool.vhd")
lib_mccp.add_source_files("tb_module.conv_pool.vhd")

lib_fcp = vu.add_library("fast_conv_pool")
lib_fcp.add_source_files("package.conv_pool.vhd")
lib_fcp.add_source_files("dp_bram.vhd")
lib_fcp.add_source_files("module.conv_pool_fast.vhd")
lib_fcp.add_source_files("tb_module.conv_pool_fast.vhd")
# Add this line to enable VCD export
vu.set_sim_option("ghdl.sim_flags", ["--vcd=wave.vcd"])
#vu.set_compile_option("ghdl.a_flags", ["--std=08"])

vu.main()
