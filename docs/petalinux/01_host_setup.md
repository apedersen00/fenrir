---
layout: default
title: 1. Host Setup and Vivado Preparation
parent: PetaLinux
nav_order: 1
---

# 1. Host Setup & Vivado Preparation

This page covers everything you need to do *before* starting the PetaLinux project itself. It's the one-time setup for your development machine and the hardware definition from Vivado.

## Setting Up the Build Host

Your build host is the computer where you will build the PetaLinux project. We recommend using Windows Subsystem for Linux (WSL) with Ubuntu 24.04.

### Install WSL and Ubuntu

1.  Open a Command Prompt or PowerShell window in Windows and install WSL with Ubuntu:
    ```bash
    wsl --install ubuntu
    ```
2.  Launch your new Ubuntu distribution:
    ```bash
    wsl -d ubuntu
    ```
3.  Upon first launch, it's a good practice to update and upgrade the system packages:
    ```bash
    sudo apt update && sudo apt upgrade
    ```

### Install PetaLinux Tools

1.  Download the PetaLinux environment setup script (`plnx-env-setup.sh`) from the AMD/Xilinx support website. This script will help install all the required dependencies.
2.  Create a directory for the PetaLinux tools:
    ```bash
    mkdir -p ~/petalinux-tools
    ```
3.  Download and install the PetaLinux tools installer from the [official Xilinx downloads page](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-design-tools.html) into the `~/petalinux-tools` directory. You will need to run the installer with `sudo`.

## Vivado Hardware Configuration

Before creating the PetaLinux project, you need a hardware definition file (`.xsa`) exported from Vivado.

### Processing System Configuration

In your Vivado project, ensure that the Zynq Processing System is configured correctly:
* **Interfaces**: The Ethernet, USB0, and UART1 interfaces must be configured **exactly** as they are in a default Digilent Zybo project.
* **MIO & Clocks**: All MIO (Multiplexed I/O) and clock configurations should be identical to the Zybo defaults.

### Generate Bitstream and Export Hardware

1.  In Vivado, run synthesis and implementation to generate the bitstream.
2.  Go to `File > Export > Export Hardware`. Make sure to include the bitstream in the exported hardware file.
3.  Copy the generated `.xsa` file to your `~/petalinux-tools` directory on the build host.