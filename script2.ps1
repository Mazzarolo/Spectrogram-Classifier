$repeticoes = 20

for ($i = 1; $i -le $repeticoes; $i++) {
    Write-Host "Execução #$i"
    python main.py 5 1015 0
}
