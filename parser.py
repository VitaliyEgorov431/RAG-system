import pymupdf4llm
import os

# --- НАСТРОЙКИ ---
# Укажи точное имя твоего файла
PDF_PATH = "K:/document1.pdf" 
# Имя файла, куда сохраним результат
OUTPUT_FILE = "document1.md"

# Проверка, что файл существует
if not os.path.exists(PDF_PATH):
    print(f"ОШИБКА: Файл '{PDF_PATH}' не найден. Проверь название.")
else:
    print(f"--- Начинаю чтение файла: {PDF_PATH} ---")
    
    try:
        # 1. Основная магия: превращаем PDF в Markdown
        # Эта функция сама находит таблицы и рисует их символами "|"
        md_text = pymupdf4llm.to_markdown(PDF_PATH)
        
        # 2. Сохраняем результат в файл, чтобы ты мог посмотреть
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(md_text)
            
        print(f"УСПЕХ! Текст сохранен в файл: {OUTPUT_FILE}")
        print("Теперь открой этот файл в Блокноте или VS Code и проверь таблицы.")
        
    except Exception as e:
        print(f"Произошла ошибка при чтении: {e}")