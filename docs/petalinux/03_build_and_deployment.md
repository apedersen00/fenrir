---
layout: default
title: 3. Build & Deployment
parent: PetaLinux
nav_order: 3
---

# 3. Building and Deploying the Image

This page walks you through building the configured project and deploying the resulting images to an SD card for booting on the hardware.

### Build the PetaLinux Project

1.  Run the build command. This process can take a significant amount of time.
    ```bash
    petalinux-build
    ```
2.  If you encounter errors, you can perform a clean build:
    ```bash
    petalinux-build -x mrproper
    petalinux-build
    ```
3.  After a successful build, create the boot image:
    ```bash
    petalinux-package --boot --fsbl --fpga --u-boot
    ```

### Prepare the SD Card

You need an SD card with two partitions: a FAT32 partition for boot files and an EXT4 partition for the root filesystem.

Insert the SD card in your computer. If you are using WSL, `usbipd` must be used for passthrough ([link](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)). Identify the SD card by
```bash
dmesg | tail
```

The necessary partitions can be created with `fdisk`:

```plaintext
sudo fdisk /dev/sdX    # USE ACTUAL DEVICE ID (DOUBLE CHECK!)

# inside fdisk

# d (delete existing partitions)
# n (new partition)
# p (primary)
# 1 (partition number 1)
# <Enter> (default first sector)
# +256M (size for boot partition)

# n (new partition)
# p (primary)
# 2 (partition number 2)
# <Enter> (default first sector)
# <Enter> (default last sector - uses remaining space)

# w (write changes and exit)
```

> ⚠️ **Warning**: Replace `/dev/sdX` with the actual device identifier for your SD card. Using the wrong identifier can result in data loss.

We must now format the partitions. they should exist in `/dev` as `/dev/sdX1` and `/dev/sdX2`:

```bash
sudo mkfs.vfat /dev/sdX1
sudo mkfs.ext4 /dev/sdX2
```

### Copy Files to the SD Card

1.  **Mount the partitions**:
    ```bash
    sudo mkdir -p /mnt/boot /mnt/rootfs
    sudo mount /dev/sdX1 /mnt/boot
    sudo mount /dev/sdX2 /mnt/rootfs
    ```
2.  **Copy boot files** to the FAT32 partition:
    ```bash
    cd ~/fenrir-petalinux/images/linux
    sudo cp BOOT.BIN boot.scr image.ub /mnt/boot/
    ```
3.  **Extract the root filesystem** to the EXT4 partition:
    ```bash
    sudo tar -xzvf rootfs.tar.gz -C /mnt/rootfs
    ```
4.  **Sync and unmount**:
    ```bash
    sync
    sudo umount /mnt/boot /mnt/rootfs
    ```

### Boot the Device

1.  Insert the prepared SD card into the Zybo board.
2.  Set the boot mode jumper to **SD**.
3.  Connect a USB cable for power and a serial console (Baud rate: 115200).
4.  Power on the board. You should see boot messages in your serial terminal.
