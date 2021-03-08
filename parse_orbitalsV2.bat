@echo off

setlocal enabledelayedexpansion

set IN_FOLDER=%1
set OUT_FOLDER=%2

set PARSER="D:\Projects\y4-project\y4_python\parse_orbitalsV2.py"

set /a INC=1
for %%f in (%IN_FOLDER%\*.log) do (
    > %OUT_FOLDER%\%%~nf.json (
        "C:\Python38\python.exe" %PARSER% -i %%f --orbitals "homo,lumo"
    )
    echo !INC!
    set /a INC +=1
)