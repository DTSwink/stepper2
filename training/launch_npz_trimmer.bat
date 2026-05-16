@echo off
set "ROOT=%~dp0.."
start "" "%ROOT%\.tools\python310\pythonw.exe" "%ROOT%\training\npz_trimmer_app.py" %*
