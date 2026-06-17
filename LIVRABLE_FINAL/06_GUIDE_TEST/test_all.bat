@echo off
setlocal

echo ==========================================
echo Test rapide du projet Deep Learning EMSI
echo ==========================================
echo.

cd /d "%~dp0\..\.."

echo [1/3] Verification de l'aide du script principal
python main.py --help
if errorlevel 1 goto :error
echo.

echo [2/3] Lancement du quick-run complet
python main.py --task all --quick-run
if errorlevel 1 goto :error
echo.

echo [3/3] Verification du dossier artifacts
if exist "artifacts\partie_1_mlp" (
  echo OK - artifacts\partie_1_mlp detecte
) else (
  echo ATTENTION - artifacts\partie_1_mlp introuvable
)

if exist "artifacts\partie_2_cnn" (
  echo OK - artifacts\partie_2_cnn detecte
) else (
  echo ATTENTION - artifacts\partie_2_cnn introuvable
)

if exist "artifacts\partie_3_sequences" (
  echo OK - artifacts\partie_3_sequences detecte
) else (
  echo ATTENTION - artifacts\partie_3_sequences introuvable
)

echo.
echo Test termine.
pause
exit /b 0

:error
echo.
echo Une erreur est survenue pendant le test.
echo Verifie d'abord l'installation des dependances Python.
pause
exit /b 1
