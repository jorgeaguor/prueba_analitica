REM Definir el nombre del entorno virtual y el archivo de texto con las librerías
REM Crear el entorno virtual
echo Creando entorno virtual...
python -m venv .venv

REM Activar el entorno virtual (en Windows)
echo Activando entorno virtual...
call .venv\Scripts\activate

python -m pip install --upgrade pip

REM Instalar las librerías desde el archivo de texto
echo Instalando librerias...
pip install -r requirements.txt

REM Abrir Visual Studio Code con el entorno virtual activado 
echo Abriendo Visual Studio Code...
code .

REM Pausa para ver cualquier mensaje de error (opcional) pyinstaller --onefile --icon=6011_delete_user_icon.ico Main_MACD.py
pause