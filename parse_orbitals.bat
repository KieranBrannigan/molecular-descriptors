@echo off

set IN_FOLDER=%1
set OUT_FOLDER=%2

set PARSER="D:\Projects\y4-project\python\parse_orbitals.py"

for %%f in (%IN_FOLDER%\*.log) do (
    > %OUT_FOLDER%\%%~nf.json (
        "C:\Python38\python.exe" %PARSER% -i %%f --orbitals "homo"
    )
)