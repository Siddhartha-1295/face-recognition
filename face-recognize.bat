@ECHO OFF

SET "PYTHONHOME=C:\Users\adithya\AppData\Local\Programs\Python\Python36-32"

pause
ECHO Training the Faces
%PYTHONHOME%\python.exe %PYTHONHOME%\src\faces-train.py
IF %ERRORLEVEL% EQU 0 ( 
   ECHO  Executing the Face Recognition
   pause
   %PYTHONHOME%\python.exe %PYTHONHOME%\src\faces.py
   pause
)