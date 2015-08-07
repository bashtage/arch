@echo off
Setlocal EnableDelayedExpansion
REM Get current directory
SET CURRENT_WORKING_DIR=%~dp0
REM Python and NumPy versions
set VERSION=2.0
set PY_VERSION=27 33 34
set NPY_VERSION=18 19


(for %%P in (%PY_VERSION%) do (
    (for %%N in (%NPY_VERSION%) do (

        REM Trick to force a delay. Windows sometimes has issues with rapid file deletion
        call PING 1.1.1.1 -n 1 -w 5000 >NUL
        REM Clean up
        robocopy C:\Anaconda\conda-bld\work\ c:\temp\conda-work-trash * /MOVE /S /NFL /NP
        robocopy C:\Anaconda\envs\_build\ c:\temp\conda-build-trash * /MOVE /S /NFL /NP
        del C:\Anaconda\conda-bld\_build\*.*? /s
        rd /s /q C:\Anaconda\envs\_build
        del /q c:\temp\conda-work-trash\*.*? /s
        del /q c:\temp\conda-build-trash\*.*? /s
        rmdir C:\Anaconda\conda-bld\work\.git /S /Q
        REM Trick to force a delay. Windows sometimes has issues with rapid file deletion
        call PING 1.1.1.1 -n 1 -w 30000 >NUL

        cd %CURRENT_WORKING_DIR%
        IF %%P==27 (
            REM No longer needed using Python Tools
            REM call python2_setup.bat
        ) ELSE (
            call python3_setup.bat
        )
        set CONDA_PY=%%P
        set CONDA_NPY=%%N
        echo Python: !CONDA_PY!, NumPy: !CONDA_NPY!

        REM Remove from binstar
        binstar remove bashtage/arch/!VERSION!/win-64\arch-!VERSION!-np!CONDA_NPY!py!CONDA_PY!_0.tar.bz2 --force
        cd %CURRENT_WORKING_DIR%
        conda build binstar

        REM Trick to force a delay. Windows sometimes has issues with rapid file deletion
        call PING 1.1.1.1 -n 1 -w 5000 >NUL
        rd /s /q C:\Anaconda\envs\_build
        ))
)) 
