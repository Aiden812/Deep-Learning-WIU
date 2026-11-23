@echo off
rem Launch SmileCheck SG (Expo): builds the web app if needed, then starts the screening server
rem (web app + API) on this computer and the local network.
cd /d "%~dp0"
if not exist node_modules call npm install
if not exist dist\index.html call npx expo export -p web
cd /d "%~dp0..\.."
python "prototype\Daniel Expo\server.py"
pause
