@echo off
setlocal

:: === Hardcoded Variables ===
set "PLATFORM_DIR=fenrir_zybo"
set "APP_DIR=hello_world"

set "SRC_DIR=.\vitis\%PLATFORM_DIR%\hw\sdt"
set "DST_DIR=.\vitis\%APP_DIR%\_ide\bitstream"

:: Find and copy the first .bit file
for %%F in ("%SRC_DIR%\*.bit") do (
    echo Found bitstream: %%~nxF
    echo Copying to: %DST_DIR%\%%~nxF

    :: Create destination directory if it doesn't exist
    if not exist "%DST_DIR%" (
        mkdir "%DST_DIR%"
    )

    copy /Y "%%F" "%DST_DIR%\%%~nxF"
    echo Copy successful.
    goto :eof
)

echo No .bit file found in %SRC_DIR%
exit /b 1
