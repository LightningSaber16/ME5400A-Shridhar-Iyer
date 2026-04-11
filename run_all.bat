@echo off
setlocal

:: ─────────────────────────────────────────────────────────────────────────────
::  run_all.bat  —  Full pipeline: train → evaluate → compare
::  Place this file in your project folder and double-click, or run from cmd.
::  All steps run in order. If any step fails the batch stops immediately.
:: ─────────────────────────────────────────────────────────────────────────────

echo.
echo ============================================================
echo  ME5400A — Braitenberg RL Pipeline
echo ============================================================
echo.

:: ── Step 1: Train Stage 2 (RL on original base) ──────────────────────────────
echo [1/6] Training Stage 2 — RL on original base (2000 episodes)...
python train_rl.py --episodes 1000 --out models
if %ERRORLEVEL% neq 0 ( echo ERROR in Stage 2 training. Stopping. & exit /b 1 )
echo Stage 2 training complete.
echo.

:: ── Step 2: Train Stage 3 (RL on improved base) ──────────────────────────────
echo [2/6] Training Stage 3 — RL on improved base (2000 episodes)...
python train_rl.py --improved --episodes 1000 --out models
if %ERRORLEVEL% neq 0 ( echo ERROR in Stage 3 training. Stopping. & exit /b 1 )
echo Stage 3 training complete.
echo.

:: ── Step 3: Evaluate Stage 1 (pure Braitenberg) ──────────────────────────────
echo [3/6] Evaluating Stage 1 — Pure Braitenberg...
python run_experiments.py
if %ERRORLEVEL% neq 0 ( echo ERROR in Stage 1 evaluation. Stopping. & exit /b 1 )
echo Stage 1 evaluation complete.
echo.

:: ── Step 4: Evaluate Stage 2 (RL on original base) ───────────────────────────
echo [4/6] Evaluating Stage 2 — RL on original base...
python run_hybrid_experiments.py --policy models/policy_final
if %ERRORLEVEL% neq 0 ( echo ERROR in Stage 2 evaluation. Stopping. & exit /b 1 )
echo Stage 2 evaluation complete.
echo.

:: ── Step 5: Evaluate Stage 3 (RL on improved base) ───────────────────────────
echo [5/6] Evaluating Stage 3 — RL on improved base...
python run_improved_experiments.py --policy models/policy_improved_final
if %ERRORLEVEL% neq 0 ( echo ERROR in Stage 3 evaluation. Stopping. & exit /b 1 )
echo Stage 3 evaluation complete.
echo.

:: ── Step 6: Evaluate Stage 4 (improved base, fixed weights) ──────────────────
echo [6/6] Evaluating Stage 4 — Improved base, fixed weights...
python run_improved_baseline_experiments.py
if %ERRORLEVEL% neq 0 ( echo ERROR in Stage 4 evaluation. Stopping. & exit /b 1 )
echo Stage 4 evaluation complete.
echo.

:: ── Step 7: Generate comparison plots ────────────────────────────────────────
echo [7/7] Generating comparison plots...
python compare.py
if %ERRORLEVEL% neq 0 ( echo ERROR in compare.py. Stopping. & exit /b 1 )
echo Plots generated.
echo.

echo ============================================================
echo  All steps completed successfully.
echo ============================================================
echo.
pause
