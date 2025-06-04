#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> // For uint32_t
#include <fcntl.h>    // For O_RDWR
#include <sys/mman.h> // For mmap, munmap
#include <unistd.h>   // For close, getpagesize

// Define the physical base address of your AXI GPIO peripheral
#define AXI_GPIO_BASE 0x41210000

// Define the size of the memory region for your peripheral (64KB as per device tree)
#define AXI_GPIO_SIZE 0x10000

// Define offsets for AXI GPIO registers (common for Xilinx AXI GPIO)
#define GPIO_DATA_OFFSET 0x0  // Data Register
#define GPIO_TRI_OFFSET  0x4  // Tri-state Control Register (Direction)

// Define the 32-bit value you want to write to the GPIO_DATA register
#define VALUE_TO_WRITE_GPIO_DATA 0x00000003 // Example: set lower 2 bits high
// Define the value to write to GPIO_TRI to set pins as output
#define VALUE_FOR_GPIO_TRI_OUTPUT 0x00000000 // All pins as output

int main() {
    int fd;
    void *mapped_base_gpio;
    volatile uint32_t *gpio_data_reg;
    volatile uint32_t *gpio_tri_reg;
    off_t target_gpio = AXI_GPIO_BASE;
    long page_size;

    // Open /dev/mem
    // Needs to be run as root
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd == -1) {
        perror("Error opening /dev/mem. Are you root?");
        return EXIT_FAILURE;
    }

    // Get the system page size
    page_size = sysconf(_SC_PAGE_SIZE);
    if (page_size == -1) {
        perror("Error getting page size");
        close(fd);
        return EXIT_FAILURE;
    }

    // Map the physical memory region for the GPIO peripheral into userspace
    off_t map_base_addr_gpio = (target_gpio / page_size) * page_size;
    off_t page_offset_gpio = target_gpio % page_size;

    mapped_base_gpio = mmap(NULL,
                            AXI_GPIO_SIZE + page_offset_gpio,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED,
                            fd,
                            map_base_addr_gpio);

    if (mapped_base_gpio == MAP_FAILED) {
        perror("mmap for GPIO failed");
        close(fd);
        return EXIT_FAILURE;
    }

    printf("Successfully mapped GPIO physical address 0x%lX to virtual address %p\n",
           (unsigned long)AXI_GPIO_BASE, mapped_base_gpio);

    // Calculate virtual addresses for GPIO_TRI and GPIO_DATA registers
    gpio_tri_reg = (volatile uint32_t *)((char *)mapped_base_gpio + page_offset_gpio + GPIO_TRI_OFFSET);
    gpio_data_reg = (volatile uint32_t *)((char *)mapped_base_gpio + page_offset_gpio + GPIO_DATA_OFFSET);

    printf("Virtual address of GPIO_TRI register (offset 0x%X): %p\n",
           GPIO_TRI_OFFSET, (void *)gpio_tri_reg);
    printf("Virtual address of GPIO_DATA register (offset 0x%X): %p\n",
           GPIO_DATA_OFFSET, (void *)gpio_data_reg);

    // 1. Set GPIO direction: Write to GPIO_TRI to set pins as output
    printf("Writing value 0x%08X to GPIO_TRI to set direction to output...\n", VALUE_FOR_GPIO_TRI_OUTPUT);
    *gpio_tri_reg = VALUE_FOR_GPIO_TRI_OUTPUT;
    printf("GPIO_TRI write complete.\n");

    // Optionally, read back GPIO_TRI to verify (if needed)
    uint32_t read_tri_value = *gpio_tri_reg;
    printf("Read back GPIO_TRI value: 0x%08X\n", read_tri_value);
    if (read_tri_value == VALUE_FOR_GPIO_TRI_OUTPUT) {
        printf("GPIO_TRI verification successful!\n");
    } else {
        printf("GPIO_TRI verification FAILED! Expected 0x%08X, got 0x%08X\n", VALUE_FOR_GPIO_TRI_OUTPUT, read_tri_value);
    }

    // 2. Write the data value to GPIO_DATA register
    printf("Writing value 0x%08X to GPIO_DATA register...\n", VALUE_TO_WRITE_GPIO_DATA);
    *gpio_data_reg = VALUE_TO_WRITE_GPIO_DATA;
    printf("GPIO_DATA write complete.\n");

    // Optionally, read back GPIO_DATA value to verify
    uint32_t read_data_value = *gpio_data_reg;
    printf("Read back GPIO_DATA value: 0x%08X\n", read_data_value);

    if (read_data_value == VALUE_TO_WRITE_GPIO_DATA) {
        printf("GPIO_DATA verification successful!\n");
    } else {
        // Note: Reading GPIO_DATA might reflect the actual pin state, which could be
        // influenced by external pull-ups/pull-downs if not all pins are driven,
        // or if some pins are inputs. For outputs, it should match what was written.
        printf("GPIO_DATA verification FAILED or value differs! Expected 0x%08X, got 0x%08X\n", VALUE_TO_WRITE_GPIO_DATA, read_data_value);
    }

    // Unmap the memory region
    if (munmap(mapped_base_gpio, AXI_GPIO_SIZE + page_offset_gpio) == -1) {
        perror("munmap for GPIO failed");
        // Continue to close fd
    }

    // Close /dev/mem
    close(fd);

    return EXIT_SUCCESS;
}