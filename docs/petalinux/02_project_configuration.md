---
layout: default
title: 2. Project Configuration
parent: PetaLinux
nav_order: 2
---

# 2. PetaLinux Project Configuration

This page is the core of the PetaLinux configuration process, containing all the `petalinux-config` steps required to tailor the system to our hardware and needs.

### Create and Configure the Project

1.  Source the PetaLinux tools environment script to set up your shell environment:
    ```bash
    source ~/petalinux-tools/settings.sh
    ```
2.  Navigate to your home directory and create a new PetaLinux project:
    ```bash
    cd ~
    petalinux-create --type project --template zynq --name fenrir-petalinux
    ```
3.  Move into the project directory and configure it with your exported hardware definition file:
    ```bash
    cd fenrir-petalinux
    petalinux-config --get-hw-description ~/petalinux-tools/your_hardware_file.xsa
    ```

### System Configuration

After running the configuration command, a menu will appear. Make the following changes:
* Navigate to `Image Packing Configuration`.
* Set `Root filesystem type` to **EXT4 (SD/eMMC/SATA/USB)**.
* Disable `Copy final images to tftpboot`.

### Install `libtinfo5` on Ubuntu 24.04

Ubuntu 24.04 does not include the `libtinfo5` library, which is required by some of the PetaLinux tools. Install it manually:
```bash
sudo apt update
wget [http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb](http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb)
sudo apt install ./libtinfo5_6.3-2ubuntu0.1_amd64.deb
```

### Kernel and U-Boot configuration

Launch the kernel configuration menu. PetaLinux typically handles _most_ the necessary configurations, so you can usually exit without making changes.
```bash
petalinux-config -c kernel
```
Launch the U-Boot configuration menu to enable SD card booting.
```bash
petalinux-config -c u-boot
```
In the configuration window:
- Navigate to `Boot options`.
- Navigate to `Boot media`.
- Enable __Support for booting from SD/EMMC__.

### Enabling USB Support

1. __Kernel Configuration__: Create a file named `user_kernel.cfg` in `project-spec/meta-user/recipes-kernel/linux/linux-xlnx/` and add the following:
```
#Mandatory for Functionality
CONFIG_USB=y
CONFIG_USB_ULPI_BUS=y
CONFIG_USB_CONN_GPIO=y
CONFIG_USB_ANNOUNCE_NEW_DEVICES=y
CONFIG_USB_OHCI_LITTLE_ENDIAN=y
CONFIG_USB_SUPPORT=y
CONFIG_USB_COMMON=y
CONFIG_USB_ARCH_HAS_HCD=y
CONFIG_USB_DEFAULT_PERSIST=y
CONFIG_USB_EHCI_HCD=y
CONFIG_USB_EHCI_ROOT_HUB_TT=y
CONFIG_USB_EHCI_PCI=y
CONFIG_USB_EHCI_HCD_PLATFORM=y
CONFIG_USB_CHIPIDEA=y
CONFIG_USB_CHIPIDEA_OF=y
CONFIG_USB_CHIPIDEA_PCI=y
CONFIG_USB_PHY=y
CONFIG_NOP_USB_XCEIV=y
CONFIG_AM335X_CONTROL_USB=y
CONFIG_AM335X_PHY_USB=y
CONFIG_USB_GPIO_VBUS=y
CONFIG_USB_ULPI=y
CONFIG_USB_ULPI_VIEWPORT=y
CONFIG_USB_CHIPIDEA_HOST=y
CONFIG_USB_HID=y
CONFIG_USB_ACM=y
CONFIG_USB_PRINTER=y
CONFIG_USB_WDM=y
CONFIG_USB_TMC=y
CONFIG_USB_STORAGE=y
CONFIG_MEDIA_USB_SUPPORT=y
CONFIG_USB_VIDEO_CLASS=y
CONFIG_USB_VIDEO_CLASS_INPUT_EVDEV=y
CONFIG_USB_GSPCA=y
CONFIG_V4L_PLATFORM_DRIVERS=y
CONFIG_VIDEO_ADV7604=y
```

2. __Append to Recipe__: Reference this new configuration file in the `linux-xlnx_%.bbappend` file located in the same directory.

```bash
FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

SRC_URI:append = " file://bsp.cfg"
KERNEL_FEATURES:append = " bsp.cfg"
SRC_URI += "file://user_2025-06-05-09-23-00.cfg \
            file://user_2025-06-05-09-59-00.cfg \
            file://user_2025-06-05-10-39-00.cfg \
            file://user_2025-06-05-12-20-00.cfg \
            file://user_kernel.cfg \
            "
```

3. __Device Tree Modification__: Open `project-spec/meta-user/recipes-bsp/device-tree/files/system-user.dtsi` and add the following:

```plaintext
/include/ "system-conf.dtsi"
/{
    usb_phy0: usb_phy@0 {
        compatible = "ulpi-phy";
        #phy-cells = <0>;
        reg = <0xe0002000 0x1000>;
        view-port = <0x0170>;
        drv-vbus;
    };
};

&usb0 {
    dr_mode = "host";
    usb-phy = <&usb_phy0>;
};
```
### RootFS Configuration

1. To add packages via a configuration file, edit `project-spec/meta-user/conf/user-rootfsconfig` and add this line:
```
CONFIG_cmake
```

2. Launch the RootFS configuration menu:
```bash
petalinux-config -c rootfs
```

3. In the menu, enable: `Filesystem packages > libs > libusb1`, `misc > packagegroup-core-buildessential`, `network > openssh`, and `User packages > cmake`, `git`, `vim`, and `dnf`.