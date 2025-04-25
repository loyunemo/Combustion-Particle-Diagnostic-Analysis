@echo off
REM 依次运行四个Python脚本

echo Running detect.py...
python .\Particle_Detect\detect.py
if %errorlevel% neq 0 (
    echo Error running detect.py
    pause
    exit /b %errorlevel%
)

echo Running split_particle_info.py...
python .\split_particle_info.py
if %errorlevel% neq 0 (
    echo Error running split_particle_info.py
    pause
    exit /b %errorlevel%
)

echo Running main.py...
python .\Particle_match\main.py
if %errorlevel% neq 0 (
    echo Error running main.py
    pause
    exit /b %errorlevel%
)

echo Running show.py...
python .\Particle_match\show.py
if %errorlevel% neq 0 (
    echo Error running show.py
    pause
    exit /b %errorlevel%
)

echo All scripts completed successfully.
pause
