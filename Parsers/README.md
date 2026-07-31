# Data collection scripts

Скрипты, которыми был собран `data/expression_utr_summary.csv`. Не входят в основной ML-пайплайн; сохранены для воспроизводимости и как часть портфолио.

## Структура

1. **`1. Collection/`** — скрейпинг UTRdb (Selenium + JSON парсинг). Извлекает пары `gene_symbol → (UTR5, UTR3)`.
2. **`2. Connection with ProteomicsDB/`** — скрейпинг ProteomicsDB (requests + BeautifulSoup). Извлекает уровни экспрессии белка по тканям.
3. **`3. Agregation/`** — объединение двух источников на уровне `gene_symbol` в итоговый CSV.

Скрипты хардкодят пути и требуют локальной установки ChromeDriver — они запускались один раз для сбора датасета. Итоговый CSV лежит в `data/expression_utr_summary.csv` и достаточен для запуска пайплайна.
