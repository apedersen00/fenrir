---
layout: default
title: 5. Troubleshooting
parent: PetaLinux
nav_order: 5
---

# 5. Troubleshooting

This page lists common issues encountered during the PetaLinux build or runtime, along with their solutions.

### PetaLinux Build Error: AXI IP Makefile Syntax

For some _insane_ reason, the auto-generated Makefile for a custom AXI IP core can contain syntax errors. This prevents the PetaLinux build from compiling the driver for that IP.

**Problem**: The original Makefile uses incorrect syntax for variable expansion and object file definition.

#### Defective Makefile

```makefile
COMPILER=
ARCHIVER=
CP=cp
COMPILER_FLAGS=
EXTRA_COMPILER_FLAGS=
LIB=libxil.a

RELEASEDIR=../../../lib
INCLUDEDIR=../../../include
INCLUDES=-I./. -I${INCLUDEDIR}

INCLUDEFILES=$(wildcard *.h)
LIBSOURCES=($wildcard *.c)
OUTS = ($wildcard *.o)

libs:
	echo "Compiling fenrir_axi..."
	$(COMPILER) $(COMPILER_FLAGS) $(EXTRA_COMPILER_FLAGS) $(INCLUDES) $(LIBSOURCES)
	$(ARCHIVER) -r ${RELEASEDIR}/${LIB} ${OUTS}
	make clean

include:
	${CP} $(INCLUDEFILES) $(INCLUDEDIR)

clean:
	rm -rf ${OUTS}
```

#### Corrected Makefile

```makefile
COMPILER=
ARCHIVER=
CP=cp
COMPILER_FLAGS=
EXTRA_COMPILER_FLAGS=
LIB=libxil.a

RELEASEDIR=../../../lib
INCLUDEDIR=../../../include
INCLUDES=-I./. -I${INCLUDEDIR}

INCLUDEFILES=$(wildcard *.h)
LIBSOURCES=$(wildcard *.c)
OBJS=$(LIBSOURCES:.c=.o)

libs:
	@echo "Compiling fenrir_axi sources: $(LIBSOURCES)"
	$(COMPILER) $(COMPILER_FLAGS) $(EXTRA_COMPILER_FLAGS) $(INCLUDES) $(LIBSOURCES)

	@echo "Archiving object files $(OBJS) into ${RELEASEDIR}/${LIB}"
	# This command adds the compiled object files to the library.
	$(ARCHIVER) -r ${RELEASEDIR}/${LIB} $(OBJS)

	@echo "Running clean target from within libs target..."
	# Recursively call the 'clean' target. Use $(MAKE) for recursive calls.
	$(MAKE) clean

include:
	@echo "Copying header files $(INCLUDEFILES) to $(INCLUDEDIR)"
	${CP} $(INCLUDEFILES) $(INCLUDEDIR)

clean:
	@echo "Cleaning object files: $(OBJS)"
	# Use -f to suppress errors if files don't exist (e.g., on a truly clean run)
	rm -f $(OBJS)
```

#### How to Apply the Fix

1. Locate the faulty Makefile within your IP core's repository in the Vivado project.

2. Replace its content with the corrected version above.

3. In Vivado, repackage your IP.

4. Re-run synthesis and implementation.

5. Export the new hardware (`.xsa`) file and re-import it into your PetaLinux project (`petalinux-config --get-hw-description`).

Perform a clean build of your PetaLinux project (`petalinux-build -x mrproper && petalinux-build`).