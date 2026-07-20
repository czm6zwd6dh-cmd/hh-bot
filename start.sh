#!/bin/bash

# Цвета для красоты
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== HH-Bot Launcher ===${NC}"

# Проверка .env
if [ ! -f .env ]; then
    echo "❌ Файл .env не найден! Создай его из .env.example"
    exit 1
fi

# Загрузка переменных
export $(grep -v '^#' .env | xargs)

# Проверка TELEGRAM_TOKEN
if [ -z "$TELEGRAM_TOKEN" ]; then
    echo "❌ TELEGRAM_TOKEN не задан в .env"
    exit 1
fi

echo -e "${GREEN}✓ Переменные окружения загружены${NC}"
echo -e "${GREEN}✓ Запуск бота...${NC}"

python app.py
