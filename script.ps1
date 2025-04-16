# Array de tamanhos da janela
$window_sizes = @(40, 80, 160, 320)

# Loop para cada tamanho de janela
foreach ($window_size in $window_sizes) {
    # Loop para calcular os valores de "start"
    for ($start = 7; $start -le 1015; $start += $window_size) {
        # Verifica se start + window_size excede 1015
        if (($start + $window_size) -gt 1015) {
            continue
        }

        for ($repeat = 1; $repeat -le 1; $repeat++) {
            Write-Host "Executando com window_size=$window_size e start=$start"
            python main.py 5 $window_size $start
        }
    }
}