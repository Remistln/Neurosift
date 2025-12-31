@echo off
echo Starting NeuroSift...

call "%~dp0venv\Scripts\activate.bat"
"%~dp0venv\Scripts\python.exe" -m streamlit run src/app/main.py

pause
