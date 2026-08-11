@echo off
setlocal

set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"

if not exist "%VSDEVCMD%" (
  echo Could not find Visual Studio 2022 VsDevCmd.bat.
  echo Install Visual Studio C++ Build Tools or add CMake/a C++17 compiler to PATH.
  exit /b 1
)

if not exist build mkdir build

call "%VSDEVCMD%" -arch=x64 >nul
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /EHsc /O2 /W4 /Fe:build\gcr_benchmark.exe src\gcr_benchmark.cpp
if errorlevel 1 exit /b 1

echo Built build\gcr_benchmark.exe
