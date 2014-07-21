REM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
REM Batch file to build binstar binaries from statsmodel head for Python 2.7,
REM 3.3 and 3.4
REM
REM Assumes that
REM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

REM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
REM Python 2.7
REM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

REM Get current directory
SET CURRENT_WORKING_DIR=%~dp0

REM Set python version
set CONDA_PY=27

REM Clean up
call PING 1.1.1.1 -n 1 -w 5000 >NUL
robocopy C:\Anaconda\conda-bld\work\ c:\temp\conda-work-trash * /MOVE /S 
del c:\temp\conda-work-trash\*.*? /s
rmdir C:\Anaconda\conda-bld\work\.git /S /Q
REM Force a delay to let it complete
call PING 1.1.1.1 -n 1 -w 60000 >NUL

REM Setup compiler
set PATH=C:\Program Files (x86)\Microsoft Visual Studio 9.0\Common7\IDE;%PATH%
set INCLUDE=C:\Program Files\Microsoft SDKs\Windows\v7.0\Include;C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\include;%INCLUDE%
set LIB=C:\Program Files\Microsoft SDKs\Windows\v7.0\Lib\x64;C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\lib\amd64
set DISTUTILS_USE_SDK=1
CALL "C:\temp\setenv" /x64 /release

REM Remove existing version
binstar remove bashtage/arch/1.0/win-64\arch-1.0-np18py27_0.tar.bz2 --force

REM Build binstar
conda build binstar


REM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
REM Python 3.3
REM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

REM Set python version
set CONDA_PY=33

REM Clean up
cd %CURRENT_WORKING_DIR%
REM Force a delay before removal
call PING 1.1.1.1 -n 1 -w 5000 >NUL
robocopy C:\Anaconda\conda-bld\work\ c:\temp\conda-work-trash * /MOVE /S 
del c:\temp\conda-work-trash\*.*? /s
rmdir C:\Anaconda\conda-bld\work\.git /S /Q
REM Force a delay to let it complete
call PING 1.1.1.1 -n 1 -w 60000 >NUL

REM Setup compiler
set PATH=C:\Program Files (x86)\Microsoft Visual Studio 10.0\Common7\IDE;%PATH%
cd C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin
set DISTUTILS_USE_SDK=1
CALL "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\setenv" /x64 /release

REM Remove existing version
binstar remove bashtage/arch/1.0/win-64\arch-1.0-np18py33_0.tar.bz2 --force

REM Build binstar
cd %CURRENT_WORKING_DIR%
conda build binstar
