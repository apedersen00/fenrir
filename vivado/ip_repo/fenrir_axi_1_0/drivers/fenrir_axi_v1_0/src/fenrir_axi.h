
#ifndef FENRIR_AXI_H
#define FENRIR_AXI_H


/****************** Include Files ********************/
#include "xil_types.h"
#include "xstatus.h"

#define FENRIR_AXI_S00_AXI_SLV_REG0_OFFSET 0
#define FENRIR_AXI_S00_AXI_SLV_REG1_OFFSET 4
#define FENRIR_AXI_S00_AXI_SLV_REG2_OFFSET 8
#define FENRIR_AXI_S00_AXI_SLV_REG3_OFFSET 12
#define FENRIR_AXI_S00_AXI_SLV_REG4_OFFSET 16
#define FENRIR_AXI_S00_AXI_SLV_REG5_OFFSET 20
#define FENRIR_AXI_S00_AXI_SLV_REG6_OFFSET 24
#define FENRIR_AXI_S00_AXI_SLV_REG7_OFFSET 28
#define FENRIR_AXI_S00_AXI_SLV_REG8_OFFSET 32
#define FENRIR_AXI_S00_AXI_SLV_REG9_OFFSET 36
#define FENRIR_AXI_S00_AXI_SLV_REG10_OFFSET 40
#define FENRIR_AXI_S00_AXI_SLV_REG11_OFFSET 44
#define FENRIR_AXI_S00_AXI_SLV_REG12_OFFSET 48
#define FENRIR_AXI_S00_AXI_SLV_REG13_OFFSET 52
#define FENRIR_AXI_S00_AXI_SLV_REG14_OFFSET 56
#define FENRIR_AXI_S00_AXI_SLV_REG15_OFFSET 60
#define FENRIR_AXI_S00_AXI_SLV_REG16_OFFSET 64
#define FENRIR_AXI_S00_AXI_SLV_REG17_OFFSET 68
#define FENRIR_AXI_S00_AXI_SLV_REG18_OFFSET 72
#define FENRIR_AXI_S00_AXI_SLV_REG19_OFFSET 76
#define FENRIR_AXI_S00_AXI_SLV_REG20_OFFSET 80
#define FENRIR_AXI_S00_AXI_SLV_REG21_OFFSET 84
#define FENRIR_AXI_S00_AXI_SLV_REG22_OFFSET 88
#define FENRIR_AXI_S00_AXI_SLV_REG23_OFFSET 92
#define FENRIR_AXI_S00_AXI_SLV_REG24_OFFSET 96
#define FENRIR_AXI_S00_AXI_SLV_REG25_OFFSET 100
#define FENRIR_AXI_S00_AXI_SLV_REG26_OFFSET 104
#define FENRIR_AXI_S00_AXI_SLV_REG27_OFFSET 108
#define FENRIR_AXI_S00_AXI_SLV_REG28_OFFSET 112
#define FENRIR_AXI_S00_AXI_SLV_REG29_OFFSET 116
#define FENRIR_AXI_S00_AXI_SLV_REG30_OFFSET 120
#define FENRIR_AXI_S00_AXI_SLV_REG31_OFFSET 124
#define FENRIR_AXI_S00_AXI_SLV_REG32_OFFSET 128
#define FENRIR_AXI_S00_AXI_SLV_REG33_OFFSET 132
#define FENRIR_AXI_S00_AXI_SLV_REG34_OFFSET 136
#define FENRIR_AXI_S00_AXI_SLV_REG35_OFFSET 140
#define FENRIR_AXI_S00_AXI_SLV_REG36_OFFSET 144
#define FENRIR_AXI_S00_AXI_SLV_REG37_OFFSET 148
#define FENRIR_AXI_S00_AXI_SLV_REG38_OFFSET 152
#define FENRIR_AXI_S00_AXI_SLV_REG39_OFFSET 156
#define FENRIR_AXI_S00_AXI_SLV_REG40_OFFSET 160
#define FENRIR_AXI_S00_AXI_SLV_REG41_OFFSET 164
#define FENRIR_AXI_S00_AXI_SLV_REG42_OFFSET 168
#define FENRIR_AXI_S00_AXI_SLV_REG43_OFFSET 172
#define FENRIR_AXI_S00_AXI_SLV_REG44_OFFSET 176
#define FENRIR_AXI_S00_AXI_SLV_REG45_OFFSET 180
#define FENRIR_AXI_S00_AXI_SLV_REG46_OFFSET 184
#define FENRIR_AXI_S00_AXI_SLV_REG47_OFFSET 188
#define FENRIR_AXI_S00_AXI_SLV_REG48_OFFSET 192
#define FENRIR_AXI_S00_AXI_SLV_REG49_OFFSET 196
#define FENRIR_AXI_S00_AXI_SLV_REG50_OFFSET 200
#define FENRIR_AXI_S00_AXI_SLV_REG51_OFFSET 204
#define FENRIR_AXI_S00_AXI_SLV_REG52_OFFSET 208
#define FENRIR_AXI_S00_AXI_SLV_REG53_OFFSET 212
#define FENRIR_AXI_S00_AXI_SLV_REG54_OFFSET 216
#define FENRIR_AXI_S00_AXI_SLV_REG55_OFFSET 220
#define FENRIR_AXI_S00_AXI_SLV_REG56_OFFSET 224
#define FENRIR_AXI_S00_AXI_SLV_REG57_OFFSET 228
#define FENRIR_AXI_S00_AXI_SLV_REG58_OFFSET 232
#define FENRIR_AXI_S00_AXI_SLV_REG59_OFFSET 236
#define FENRIR_AXI_S00_AXI_SLV_REG60_OFFSET 240
#define FENRIR_AXI_S00_AXI_SLV_REG61_OFFSET 244
#define FENRIR_AXI_S00_AXI_SLV_REG62_OFFSET 248
#define FENRIR_AXI_S00_AXI_SLV_REG63_OFFSET 252
#define FENRIR_AXI_S00_AXI_SLV_REG64_OFFSET 256
#define FENRIR_AXI_S00_AXI_SLV_REG65_OFFSET 260
#define FENRIR_AXI_S00_AXI_SLV_REG66_OFFSET 264
#define FENRIR_AXI_S00_AXI_SLV_REG67_OFFSET 268
#define FENRIR_AXI_S00_AXI_SLV_REG68_OFFSET 272
#define FENRIR_AXI_S00_AXI_SLV_REG69_OFFSET 276
#define FENRIR_AXI_S00_AXI_SLV_REG70_OFFSET 280
#define FENRIR_AXI_S00_AXI_SLV_REG71_OFFSET 284
#define FENRIR_AXI_S00_AXI_SLV_REG72_OFFSET 288
#define FENRIR_AXI_S00_AXI_SLV_REG73_OFFSET 292
#define FENRIR_AXI_S00_AXI_SLV_REG74_OFFSET 296
#define FENRIR_AXI_S00_AXI_SLV_REG75_OFFSET 300
#define FENRIR_AXI_S00_AXI_SLV_REG76_OFFSET 304
#define FENRIR_AXI_S00_AXI_SLV_REG77_OFFSET 308
#define FENRIR_AXI_S00_AXI_SLV_REG78_OFFSET 312
#define FENRIR_AXI_S00_AXI_SLV_REG79_OFFSET 316
#define FENRIR_AXI_S00_AXI_SLV_REG80_OFFSET 320
#define FENRIR_AXI_S00_AXI_SLV_REG81_OFFSET 324
#define FENRIR_AXI_S00_AXI_SLV_REG82_OFFSET 328
#define FENRIR_AXI_S00_AXI_SLV_REG83_OFFSET 332
#define FENRIR_AXI_S00_AXI_SLV_REG84_OFFSET 336
#define FENRIR_AXI_S00_AXI_SLV_REG85_OFFSET 340
#define FENRIR_AXI_S00_AXI_SLV_REG86_OFFSET 344
#define FENRIR_AXI_S00_AXI_SLV_REG87_OFFSET 348
#define FENRIR_AXI_S00_AXI_SLV_REG88_OFFSET 352
#define FENRIR_AXI_S00_AXI_SLV_REG89_OFFSET 356
#define FENRIR_AXI_S00_AXI_SLV_REG90_OFFSET 360
#define FENRIR_AXI_S00_AXI_SLV_REG91_OFFSET 364
#define FENRIR_AXI_S00_AXI_SLV_REG92_OFFSET 368
#define FENRIR_AXI_S00_AXI_SLV_REG93_OFFSET 372
#define FENRIR_AXI_S00_AXI_SLV_REG94_OFFSET 376
#define FENRIR_AXI_S00_AXI_SLV_REG95_OFFSET 380
#define FENRIR_AXI_S00_AXI_SLV_REG96_OFFSET 384
#define FENRIR_AXI_S00_AXI_SLV_REG97_OFFSET 388
#define FENRIR_AXI_S00_AXI_SLV_REG98_OFFSET 392
#define FENRIR_AXI_S00_AXI_SLV_REG99_OFFSET 396
#define FENRIR_AXI_S00_AXI_SLV_REG100_OFFSET 400
#define FENRIR_AXI_S00_AXI_SLV_REG101_OFFSET 404
#define FENRIR_AXI_S00_AXI_SLV_REG102_OFFSET 408
#define FENRIR_AXI_S00_AXI_SLV_REG103_OFFSET 412
#define FENRIR_AXI_S00_AXI_SLV_REG104_OFFSET 416
#define FENRIR_AXI_S00_AXI_SLV_REG105_OFFSET 420
#define FENRIR_AXI_S00_AXI_SLV_REG106_OFFSET 424
#define FENRIR_AXI_S00_AXI_SLV_REG107_OFFSET 428
#define FENRIR_AXI_S00_AXI_SLV_REG108_OFFSET 432
#define FENRIR_AXI_S00_AXI_SLV_REG109_OFFSET 436
#define FENRIR_AXI_S00_AXI_SLV_REG110_OFFSET 440
#define FENRIR_AXI_S00_AXI_SLV_REG111_OFFSET 444
#define FENRIR_AXI_S00_AXI_SLV_REG112_OFFSET 448
#define FENRIR_AXI_S00_AXI_SLV_REG113_OFFSET 452
#define FENRIR_AXI_S00_AXI_SLV_REG114_OFFSET 456
#define FENRIR_AXI_S00_AXI_SLV_REG115_OFFSET 460
#define FENRIR_AXI_S00_AXI_SLV_REG116_OFFSET 464
#define FENRIR_AXI_S00_AXI_SLV_REG117_OFFSET 468
#define FENRIR_AXI_S00_AXI_SLV_REG118_OFFSET 472
#define FENRIR_AXI_S00_AXI_SLV_REG119_OFFSET 476
#define FENRIR_AXI_S00_AXI_SLV_REG120_OFFSET 480
#define FENRIR_AXI_S00_AXI_SLV_REG121_OFFSET 484
#define FENRIR_AXI_S00_AXI_SLV_REG122_OFFSET 488
#define FENRIR_AXI_S00_AXI_SLV_REG123_OFFSET 492
#define FENRIR_AXI_S00_AXI_SLV_REG124_OFFSET 496
#define FENRIR_AXI_S00_AXI_SLV_REG125_OFFSET 500
#define FENRIR_AXI_S00_AXI_SLV_REG126_OFFSET 504
#define FENRIR_AXI_S00_AXI_SLV_REG127_OFFSET 508


/**************************** Type Definitions *****************************/
/**
 *
 * Write a value to a FENRIR_AXI register. A 32 bit write is performed.
 * If the component is implemented in a smaller width, only the least
 * significant data is written.
 *
 * @param   BaseAddress is the base address of the FENRIR_AXIdevice.
 * @param   RegOffset is the register offset from the base to write to.
 * @param   Data is the data written to the register.
 *
 * @return  None.
 *
 * @note
 * C-style signature:
 * 	void FENRIR_AXI_mWriteReg(u32 BaseAddress, unsigned RegOffset, u32 Data)
 *
 */
#define FENRIR_AXI_mWriteReg(BaseAddress, RegOffset, Data) \
  	Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))

/**
 *
 * Read a value from a FENRIR_AXI register. A 32 bit read is performed.
 * If the component is implemented in a smaller width, only the least
 * significant data is read from the register. The most significant data
 * will be read as 0.
 *
 * @param   BaseAddress is the base address of the FENRIR_AXI device.
 * @param   RegOffset is the register offset from the base to write to.
 *
 * @return  Data is the data from the register.
 *
 * @note
 * C-style signature:
 * 	u32 FENRIR_AXI_mReadReg(u32 BaseAddress, unsigned RegOffset)
 *
 */
#define FENRIR_AXI_mReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))

/************************** Function Prototypes ****************************/
/**
 *
 * Run a self-test on the driver/device. Note this may be a destructive test if
 * resets of the device are performed.
 *
 * If the hardware system is not built correctly, this function may never
 * return to the caller.
 *
 * @param   baseaddr_p is the base address of the FENRIR_AXI instance to be worked on.
 *
 * @return
 *
 *    - XST_SUCCESS   if all self-test code passed
 *    - XST_FAILURE   if any self-test code failed
 *
 * @note    Caching must be turned off for this function to work.
 * @note    Self test may fail if data memory and device are not on the same bus.
 *
 */
XStatus FENRIR_AXI_Reg_SelfTest(void * baseaddr_p);

#endif // FENRIR_AXI_H
