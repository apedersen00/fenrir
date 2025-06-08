#
# build.tcl: tcl script for recreating the Vivado project 'fenrir'
#
#*****************************************************************************************

# Set the reference directory for source file relative paths (by default the value is script directory path)
set origin_dir [file dirname [info script]]

variable script_file
set script_file "build.tcl"

# Help information for this script
proc help {} {
  variable script_file
  puts "\nDescription:"
  puts "Recreate a Vivado project from this script. The created project will be"
  puts "functionally equivalent to the original project for which this script was"
  puts "generated. The script contains commands for creating a project, filesets,"
  puts "runs, adding/importing sources and setting properties on various objects.\n"
  puts "Syntax:"
  puts "$script_file"
  puts "$script_file -tclargs \[--origin_dir <path>\]"
  puts "$script_file -tclargs \[--help\]\n"
  puts "Usage:"
  puts "Name                   Description"
  puts "-------------------------------------------------------------------------"
  puts "\[--origin_dir <path>\]  Determine source file paths wrt this path. Default"
  puts "                       origin_dir path value is \".\", otherwise, the value"
  puts "                       that was set with the \"-paths_relative_to\" switch"
  puts "                       when this script was generated.\n"
  puts "\[--help\]               Print help information for this script"
  puts "-------------------------------------------------------------------------\n"
  exit 0
}

if { $::argc > 0 } {
  for {set i 0} {$i < [llength $::argc]} {incr i} {
    set option [string trim [lindex $::argv $i]]
    switch -regexp -- $option {
      "--origin_dir" { incr i; set origin_dir [lindex $::argv $i] }
      "--help"       { help }
      default {
        if { [regexp {^-} $option] } {
          puts "ERROR: Unknown option '$option' specified, please type '$script_file -tclargs --help' for usage info.\n"
          return 1
        }
      }
    }
  }
}

# Create project
create_project fenrir $origin_dir/fenrir -part xc7z010clg400-1

# Set the directory path for the new project
set proj_dir [get_property directory [current_project]]

# Reconstruct message rules
# None

# Set project properties
set obj [get_projects fenrir]
set_property -name "board_part" -value "digilentinc.com:zybo:part0:1.0" -objects $obj
set_property -name "default_lib" -value "xil_defaultlib" -objects $obj
set_property -name "ip_cache_permissions" -value "read write" -objects $obj
set_property -name "ip_output_repo" -value "$proj_dir/fenrir.cache/ip" -objects $obj
set_property -name "sim.ip.auto_export_scripts" -value "1" -objects $obj
set_property -name "simulator_language" -value "Mixed" -objects $obj
set_property -name "target_language" -value "VHDL" -objects $obj
set_property -name "xpm_libraries" -value "XPM_CDC XPM_FIFO XPM_MEMORY" -objects $obj

# Create 'sources_1' fileset (if not found)
if {[string equal [get_filesets -quiet sources_1] ""]} {
  create_fileset -srcset sources_1
}

# Set IP repository paths
set obj [get_filesets sources_1]
set_property "ip_repo_paths" "[file normalize "$origin_dir/ip_repo"]" $obj

# Rebuild user ip_repo's index before adding any source files
update_ip_catalog -rebuild

puts "INFO: Adding VHDL source files with specific standards..."

# 1. Define the full, normalized path to the file that should use the default VHDL standard.
#    Using 'file normalize' is critical for reliable path comparison.
set default_standard_file [file normalize "$origin_dir/../src/design_sources/fenrir_top.vhd"]

# 2. Find all VHDL files from all relevant directories using 'glob'.
set all_vhdl_sources [list]
lappend all_vhdl_sources {*}[glob [file normalize "$origin_dir/../src/design_sources/*.vhd"]]
lappend all_vhdl_sources {*}[glob [file normalize "$origin_dir/../src/design_sources/common/*.vhd"]]
lappend all_vhdl_sources {*}[glob [file normalize "$origin_dir/../src/design_sources/fc/*.vhd"]]

# Get a unique list of files, in case any glob patterns overlap.
set all_vhdl_sources [lsort -unique $all_vhdl_sources]

# 3. Separate the found files into two lists: one for VHDL 2008, and one for the default.
set vhdl_2008_list [list]
set vhdl_default_list [list]

foreach file $all_vhdl_sources {
  if {[string equal $file $default_standard_file]} {
    lappend vhdl_default_list $file
  } else {
    lappend vhdl_2008_list $file
  }
}

# 4. Process the VHDL 2008 list.
if { [llength $vhdl_2008_list] > 0 } {
  puts "INFO: Adding [llength $vhdl_2008_list] file(s) as VHDL 2008."
  # Add the files and get the returned file objects
  set added_2008_files [add_files -norecurse -fileset sources_1 $vhdl_2008_list]
  # Set the property on the newly added file objects
  set_property file_type "VHDL 2008" -objects $added_2008_files
} else {
  puts "INFO: No files found to be set as VHDL 2008."
}

# 5. Process the default standard list.
if { [llength $vhdl_default_list] > 0 } {
  puts "INFO: Adding [llength $vhdl_default_list] file(s) with the project's default VHDL standard."
  add_files -norecurse -fileset sources_1 $vhdl_default_list
}

# 6. Update the compile order for the entire project after all files have been added.
puts "INFO: Updating compile order."
update_compile_order -fileset sources_1

# --- End of VHDL section ---

# Create block design
source $origin_dir/src/bd/fenrir_system.tcl

# Generate the wrapper
set design_name [get_bd_designs]
make_wrapper -files [get_files $design_name.bd] -top -import

puts "INFO: Setting BD wrapper as top-level module."
set_property top ${design_name}_wrapper [get_filesets sources_1]

# Re-run compile order to ensure the new top is recognized
update_compile_order -fileset sources_1

# Create 'constrs_1' fileset (if not found)
if {[string equal [get_filesets -quiet constrs_1] ""]} {
  create_fileset -constrset constrs_1
}

# Set 'constrs_1' fileset object
set obj [get_filesets constrs_1]

# Add/Import constrs file and set constrs file properties
set file "[file normalize "$origin_dir/src/constraints/Zybo-Z7-Master.xdc"]"
set file_added [add_files -norecurse -fileset $obj $file]
set file "$origin_dir/src/constraints/Zybo-Z7-Master.xdc"
set file [file normalize $file]
set file_obj [get_files -of_objects [get_filesets constrs_1] [list "*$file"]]
set_property -name "file_type" -value "XDC" -objects $file_obj

# Empty (no sources present)

# Set 'constrs_1' fileset properties
set obj [get_filesets constrs_1]

# Create 'sim_1' fileset (if not found)
if {[string equal [get_filesets -quiet sim_1] ""]} {
  create_fileset -simset sim_1
}

# Set 'sim_1' fileset object
set obj [get_filesets sim_1]
# Empty (no sources present)

# Create 'synth_1' run (if not found)
if {[string equal [get_runs -quiet synth_1] ""]} {
  create_run -name synth_1 -part xc7z010clg400-1 -flow {Vivado Synthesis 2017} -strategy "Vivado Synthesis Defaults" -constrset constrs_1
} else {
  set_property strategy "Vivado Synthesis Defaults" [get_runs synth_1]
  set_property flow "Vivado Synthesis 2017" [get_runs synth_1]
}
set obj [get_runs synth_1]
set_property -name "needs_refresh" -value "1" -objects $obj

# set the current synth run
current_run -synthesis [get_runs synth_1]

# Create 'impl_1' run (if not found)
if {[string equal [get_runs -quiet impl_1] ""]} {
  create_run -name impl_1 -part xc7z010clg400-1 -flow {Vivado Implementation 2017} -strategy "Vivado Implementation Defaults" -constrset constrs_1 -parent_run synth_1
} else {
  set_property strategy "Vivado Implementation Defaults" [get_runs impl_1]
  set_property flow "Vivado Implementation 2017" [get_runs impl_1]
}
set obj [get_runs impl_1]
set_property -name "needs_refresh" -value "1" -objects $obj
set_property -name "steps.write_bitstream.args.readback_file" -value "0" -objects $obj
set_property -name "steps.write_bitstream.args.verbose" -value "0" -objects $obj

# set the current impl run
current_run -implementation [get_runs impl_1]

puts "INFO: Project created:fenrir"