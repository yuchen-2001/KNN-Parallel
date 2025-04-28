@echo off
setlocal ENABLEDELAYEDEXPANSION

:: Path to your executable
set EXE=knnInMpi.exe

:: List of processor counts to test
set PROCS=1 3 5 9 15 27 45 135

echo Running tests for %EXE%
echo ===============================

for %%P in (%PROCS%) do (
    echo.
    echo [Testing with %%P processors]...
    mpiexec -n %%P %EXE%
)

echo.
echo All tests complete.
pause
