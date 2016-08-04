@echo off
Setlocal EnableDelayedExpansion
REM Get current directory
SET CURRENT_WORKING_DIR=%~dp0
REM Python and NumPy versions
set VERSION=3.2
set PY_VERSION=27 35
set NPY_VERSION=110 111

conda config --set anaconda_upload yes

(for %%P in (%PY_VERSION%) do (
    IF %%P==27 (
        call python2_setup.bat
    ) ELSE (
        call python3_setup.bat
    )
    (for %%N in (%NPY_VERSION%) do (

        REM Trick to force a delay. Windows sometimes has issues with rapid file deletion
        REM call PING 1.1.1.1 -n 1 -w 5000 >NUL
        REM Clean up
        REM robocopy C:\Anaconda\conda-bld\work\ c:\temp\conda-work-trash * /MOVE /S /NFL /NP
        REM robocopy C:\Anaconda\envs\_build\ c:\temp\conda-build-trash * /MOVE /S /NFL /NP
        REM del C:\Anaconda\conda-bld\_build\*.*? /s
        REM rd /s /q C:\Anaconda\envs\_build
        REM del /q c:\temp\conda-work-trash\*.*? /s
        REM del /q c:\temp\conda-build-trash\*.*? /s
        REM rmdir C:\Anaconda\conda-bld\work\.git /S /Q
        REM Trick to force a delay. Windows sometimes has issues with rapid file deletion
        REM call PING 1.1.1.1 -n 1 -w 30000 >NUL

        cd %CURRENT_WORKING_DIR%
        set CONDA_PY=%%P
        set CONDA_NPY=%%N
        echo Python: !CONDA_PY!, NumPy: !CONDA_NPY!

        REM Remove from binstar
        anaconda remove bashtage/arch/!VERSION!/win-64/arch-!VERSION!-np!CONDA_NPY!py!CONDA_PY!_0.tar.bz2 --force
        conda build --numpy !CONDA_NPY! --python !CONDA_PY! binstar

        REM Trick to force a delay. Windows sometimes has issues with rapid file deletion
        REM call PING 1.1.1.1 -n 1 -w 5000 >NUL
        REM rd /s /q C:\Anaconda\envs\_build
        ))
)) 
