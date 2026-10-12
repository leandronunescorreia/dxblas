@echo off
setlocal EnableExtensions

rem Run all DXBLAS sample executables sequentially.
rem Usage:
rem   run_all_samples.bat [Debug^|Release] [build-directory]
rem Examples:
rem   run_all_samples.bat Release
rem   run_all_samples.bat Release build

set "CONFIG=%~1"
if not defined CONFIG set "CONFIG=Release"

set "BUILD_DIR=%~2"
if not defined BUILD_DIR set "BUILD_DIR=build"

set "BIN_DIR=%BUILD_DIR%\%CONFIG%"
if not exist "%BIN_DIR%\" (
    echo ERROR: Build output directory not found: "%BIN_DIR%"
    echo Build the project first, for example:
    echo   cmake --build "%BUILD_DIR%" --config %CONFIG%
    exit /b 2
)

set /a PASSED=0
set /a FAILED=0

echo.
echo DXBLAS sample runner
echo Configuration: %CONFIG%
echo Executable directory: %BIN_DIR%
echo.

rem Add or remove executable names here to match the samples built by CMake.
for %%S in (
    dxblas_gemm.exe
    dxblas_matrix_multiply_4096.exe
    dxblas_typed_operations.exe
    dxblas_gpumem_roundtrip.exe
    dxblas_gpumem_sum.exe
    dxblas_copy_tests.exe
    dxblas_benchmark_2048.exe
    dxblas_big_array.exe
    dxblas_first_step.exe
    dxblas_copy_timing.exe
) do call :RUN_SAMPLE "%%S"

echo.
echo ================================
echo Samples passed: %PASSED%
echo Samples failed: %FAILED%
echo ================================

if not "%FAILED%"=="0" exit /b 1
exit /b 0

:RUN_SAMPLE
set "SAMPLE=%~1"
if not exist "%BIN_DIR%\%SAMPLE%" (
    echo [SKIP] %SAMPLE% - executable not found
    exit /b 0
)

echo.
echo --------------------------------
echo Running %SAMPLE%
echo --------------------------------
pushd "%BIN_DIR%"
"%SAMPLE%"
set "RC=%ERRORLEVEL%"
popd

if "%RC%"=="0" (
    echo [PASS] %SAMPLE%
    set /a PASSED+=1
) else (
    echo [FAIL] %SAMPLE% - exit code %RC%
    set /a FAILED+=1
)
exit /b 0
