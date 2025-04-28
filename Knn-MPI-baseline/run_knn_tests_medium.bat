@echo off
setlocal ENABLEDELAYEDEXPANSION

:: Path to your executable
set EXE=knnInMpi.exe

:: List of processor counts to test
set PROCS=1 2 4 5 8 10 20 25 40 50 100 125 200 250 500 1000

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
