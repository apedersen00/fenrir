---
layout: default
title: 4. On-Target Development
parent: PetaLinux
nav_order: 4
---

# 4. Post-Boot and On-Target Development

Once the system has booted on the FPGA board, you can perform initial setup, install packages, and build your applications directly on the target.

### First Boot Setup

#### Ethernet Configuration

If the Ethernet connection doesn't work out of the box:
1.  Check kernel messages for hardware detection:
    ```bash
    dmesg | grep -i -E "eth|phy|macb|gem|mdio"
    ```
2.  If the hardware seems fine, request an IP address via DHCP (replace `enx...` with your actual interface name, which you can find with `ip a`):
    ```bash
    sudo udhcpc -i enx000a35001e53 -n -q -f
    ```

#### System Updates

1.  Set the correct date and time to avoid issues with package management certificates:
    ```bash
    # Example for June 8th, 2025, 2:30 PM
    sudo date 060814302025
    ```
2.  Update the system packages using `dnf`:
    ```bash
    sudo dnf update
    ```

### Building `libcaer` on Target

1.  Install build dependencies on the Zybo board:
    ```bash
    sudo dnf install libusb-1.0-devel opencv-devel
    ```
2.  Copy the `libcaer` source code from your host to the target:
    ```bash
    rsync -avz /path/to/libcaer/ petalinux@<TARGET_IP>:/home/petalinux/
    ```
3.  Build and install `libcaer` on the target:
    ```bash
    cd libcaer
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr
    make
    sudo make install
    ```

### Building a Custom DVS Application

Once `libcaer` is installed, you can compile your application using `g++`:
```bash
g++ -std=c++11 -pedantic -Wall -Wextra -O2 -o dvxplorer dvxplorer.cpp -D_DEFAULT_SOURCE=1 -lcaer
```