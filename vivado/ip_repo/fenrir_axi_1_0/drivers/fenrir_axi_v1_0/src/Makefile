# Makefile for fenrir_axi_v1_0 driver
# IMPORTANT: All command lines (recipes) under targets like 'libs:', 'include:', 'clean:'
# MUST start with a single TAB character, not spaces.

COMPILER=
ARCHIVER=
CP=cp
COMPILER_FLAGS=
EXTRA_COMPILER_FLAGS=
LIB=libxil.a

RELEASEDIR=../../../lib
INCLUDEDIR=../../../include
INCLUDES=-I./. -I${INCLUDEDIR}

# Source and object files
INCLUDEFILES=$(wildcard *.h)
LIBSOURCES=$(wildcard *.c)
OBJS=$(LIBSOURCES:.c=.o)

# Main target to compile sources and create the library archive
# This is typically called by the PetaLinux/Yocto build system for this driver.
# The COMPILER, ARCHIVER, COMPILER_FLAGS variables are usually passed in by the
# higher-level Makefile (e.g., from FSBL build).
libs:
	@echo "Compiling fenrir_axi sources: $(LIBSOURCES)"
	# This command compiles the C source files.
	# It relies on the compiler (e.g., arm-none-eabi-gcc) creating .o files
	# in the current directory when given .c files and the -c flag (which
	# should be part of COMPILER_FLAGS passed by the build system).
	$(COMPILER) $(COMPILER_FLAGS) $(EXTRA_COMPILER_FLAGS) $(INCLUDES) $(LIBSOURCES)

	@echo "Archiving object files $(OBJS) into ${RELEASEDIR}/${LIB}"
	# This command adds the compiled object files to the library.
	$(ARCHIVER) -r ${RELEASEDIR}/${LIB} $(OBJS)

	@echo "Running clean target from within libs target..."
	# Recursively call the 'clean' target. Use $(MAKE) for recursive calls.
	$(MAKE) clean

# Target to copy header files to the include directory
include:
	@echo "Copying header files $(INCLUDEFILES) to $(INCLUDEDIR)"
	${CP} $(INCLUDEFILES) $(INCLUDEDIR)

# Target to clean up compiled object files
clean:
	@echo "Cleaning object files: $(OBJS)"
	# Use -f to suppress errors if files don't exist (e.g., on a truly clean run)
	rm -f $(OBJS)
