setlocal
set PYTHONPATH=%cd%\..;%PYTHONPATH%
python param_test.py ^
    postgresql://postgres:i5gMr!Pfcdm$dn8YqhTf#$hL?jkb@localhost:5432/postgres ^
    "2021-10-30-00-00-45-trung-window_size=3.5" ^
    --window-size 3.5 ^
    --eeg 2021-10-28-11-08-51 ^
    --eeg 2021-10-28-11-45-12 ^
    --eeg 2021-10-28-11-53-11
endlocal
