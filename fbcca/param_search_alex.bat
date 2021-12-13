setlocal
set PYTHONPATH=%cd%\..;%PYTHONPATH%
python param_search.py ^
    postgresql://postgres:i5gMr!Pfcdm$dn8YqhTf#$hL?jkb@localhost:5432/postgres ^
    --subject alex ^
    --eeg 2021-07-07-17-28-18 ^
    --eeg 2021-07-30-20-50-47 ^
    --eeg 2021-08-19-19-23-53 ^
    --eeg 2021-08-19-19-40-07 ^
    --eeg 2021-08-19-19-53-46 ^
    --eeg 2021-08-19-21-46-04 ^
    --eeg 2021-08-19-21-52-46 ^
    --eeg 2021-08-19-22-00-58
endlocal
