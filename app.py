import os
import sqlite3
import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from openai import AsyncOpenAI  # <-- Используем асинхронный клиент
import httpx
import aiohttp
import random
from fastapi import FastAPI, Request
import uvicorn

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CHAT_ID = os.getenv("CHAT_ID")
SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан")

deepseek_available = False
client = None

if DEEPSEEK_API_KEY:
    try:
        # Используем асинхронный HTTP клиент
        http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", http_client=http_client)
        # Тестовый запрос
        asyncio.run(client.chat.completions.create(
            model="deepseek-chat", 
            messages=[{"role": "user", "content": "Привет"}], 
            max_tokens=5
        ))
        deepseek_available = True
        logger.info("DeepSeek подключен")
    except Exception as e:
        logger.warning(f"DeepSeek недоступен: {e}")
        client = None
else:
    logger.warning("DEEPSEEK_API_KEY не задан")

DB_PATH = "vacancies.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS sent_vacancies (
        id TEXT PRIMARY KEY, 
        title TEXT, 
        company TEXT, 
        sent_at TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS search_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        found_count INTEGER, 
        sent_count INTEGER, 
        searched_at TIMESTAMP
    )""")
    conn.commit()
    conn.close()

def is_vacancy_sent(vacancy_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM sent_vacancies WHERE id = ?", (vacancy_id,))
    result = c.fetchone() is not None
    conn.close()
    return result

def mark_vacancy_sent(vacancy_id, title, company):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO sent_vacancies VALUES (?, ?, ?, ?)", 
              (vacancy_id, title, company, datetime.now()))
    conn.commit()
    conn.close()

def log_search(found, sent):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO search_log (found_count, sent_count, searched_at) VALUES (?, ?, ?)", 
              (found, sent, datetime.now()))
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sent_vacancies")
    total_sent = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM search_log")
    total_searches = c.fetchone()[0]
    c.execute("SELECT found_count, sent_count, searched_at FROM search_log ORDER BY searched_at DESC LIMIT 5")
    recent = c.fetchall()
    conn.close()
    return total_sent, total_searches, recent

def cleanup_old_vacancies(days=30):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cutoff = datetime.now() - timedelta(days=days)
    c.execute("DELETE FROM sent_vacancies WHERE sent_at < ?", (cutoff,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted

USER_FILTERS = {
    "salary_min": 200000,
    "cities": {"Волгоград": "24", "Москва": "1", "Казань": "88", "Нижний Новгород": "66", 
               "Уфа": "99", "Астана": "160", "Баку": "100", "Минск": "1002"},
    "keywords": ["коммерческий директор", "руководитель отдела продаж", "директор по продажам", 
                 "директор филиала", "руководитель направления"],
    "industry_keywords": ["нефтепродукты", "ГСМ", "топливо", "бензин", "дизель", "мазут", 
                         "нефть", "нефтетрейдинг", "нефтебаза", "АЗС", "СПбМТСБ", "НПЗ", 
                         "нефтепереработка", "моторное топливо", "нефтяной", "oil", "petroleum", "fuel"],
    "exclude_words": ["стажёр", "intern", "junior", "1С", "QA", "тестировщик", "розница", 
                      "продавец-консультант", "FMCG", "продукты питания", "одежда", "обувь", 
                      "строительные материалы", "электроника", "IT", "финансы", "банки", 
                      "страхование", "недвижимость", "маркетинг", "реклама", "HR", "медицина", "образование"]
}

CANDIDATE_PROFILE = """
Имя: Зинченко Виктор Александрович
Возраст: 42 года
Город: Волгоград
Готов к переезду: Астана, Баку, Казань, Минск, Нижний Новгород, Уфа
Готов к командировкам: Да
Желаемая должность: Коммерческий директор / Руководитель отдела продаж / Директор филиала
Специализация: Нефтепродукты, ГСМ (дизельное топливо, бензин), B2B, опт
Тип занятости: Полная занятость
Формат: Офис/производство
Зарплата: от 200 000 ₽
Ключевой опыт:
• 20+ лет в продажах, 11 лет коммерческий директор
• Управление отделом продаж ГСМ (5 человек + 120 сотрудников)
• Закупки на СПбМТСБ и прямые контракты с НПЗ
• B2B, договоры с трейдерами и конечными покупателями
• Нефтебаза, перевалка до 30 000 тонн/мес, 29 АЗС
• Оборот отдела 1 млрд ₽
• Ж/д и автотранспорт, логистика
• Дебиторская задолженность, переговоры с первыми лицами
• 1С: Управление предприятием
• Вооружённые силы: командир взвода, ж/д перевозки
"""

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

def build_scraperapi_url(target_url, use_render=False):
    if not SCRAPERAPI_KEY:
        return target_url
    if use_render:
        return f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}&url={target_url}&render=true&premium=true"
    return f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}&url={target_url}&premium=true"

class RateLimiter:
    def __init__(self, min_delay=2.0, max_delay=5.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0
        self.lock = asyncio.Lock()
        self.consecutive_429s = 0
        self.cooldown_until = 0

    async def wait(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            if now < self.cooldown_until:
                wait_time = self.cooldown_until - now
                logger.info(f"Rate limiter: cooldown ещё {wait_time:.1f}с")
                await asyncio.sleep(wait_time)
                now = asyncio.get_event_loop().time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_delay:
                wait_time = self.min_delay - elapsed + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
            self.last_request_time = asyncio.get_event_loop().time()

    def on_429(self):
        self.consecutive_429s += 1
        cooldown = min(10 * (2 ** self.consecutive_429s), 120)
        jitter = random.uniform(5, 15)
        self.cooldown_until = asyncio.get_event_loop().time() + cooldown + jitter
        logger.warning(f"Rate limiter: 429 #{self.consecutive_429s}, cooldown {cooldown + jitter:.0f}с")

    def on_success(self):
        if self.consecutive_429s > 0:
            self.consecutive_429s = max(0, self.consecutive_429s - 1)

hh_rate_limiter = RateLimiter(min_delay=2.0, max_delay=5.0)

async def fetch_rss(session, city_id, keyword, per_page=20, retries=3):
    encoded_kw = keyword.replace(" ", "+")
    target_url = f"https://hh.ru/search/vacancy/rss?text={encoded_kw}&area={city_id}&items_on_page={per_page}"
    url = build_scraperapi_url(target_url, use_render=False)
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    for attempt in range(retries):
        await hh_rate_limiter.wait()
        try:
            timeout = aiohttp.ClientTimeout(total=45)
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                return await _handle_rss_response(resp, city_id, keyword)
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут RSS {city_id}/{keyword}, попытка {attempt+1}/{retries}")
            await asyncio.sleep(2 ** attempt + random.uniform(1, 3))
        except Exception as e:
            logger.error(f"Ошибка RSS {city_id}/{keyword}: {e}, попытка {attempt+1}/{retries}")
            await asyncio.sleep(2 ** attempt + random.uniform(1, 3))
    return []

async def _handle_rss_response(resp, city_id, keyword):
    if resp.status == 200:
        xml_text = await resp.text()
        logger.info(f"RSS {city_id}/{keyword}: получено {len(xml_text)} байт")
        hh_rate_limiter.on_success()
        return parse_rss(xml_text)
    elif resp.status in (403, 429):
        logger.warning(f"HH RSS {resp.status} для {city_id}/{keyword}")
        hh_rate_limiter.on_429()
        return []
    else:
        logger.warning(f"HH RSS {resp.status} для {city_id}/{keyword}")
        return []

async def fetch_html_fallback(session, city_id, keyword="коммерческий директор"):
    encoded_kw = keyword.replace(" ", "+")
    target_url = f"https://hh.ru/search/vacancy?text={encoded_kw}&area={city_id}&items_on_page=20"
    url = build_scraperapi_url(target_url, use_render=False)
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9",
    }
    await hh_rate_limiter.wait()
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                html = await resp.text()
                hh_rate_limiter.on_success()
                return parse_html_vacancies(html, city_id)
            elif resp.status == 429:
                hh_rate_limiter.on_429()
                return []
            else:
                return []
    except Exception as e:
        logger.error(f"HTML fallback ошибка: {e}")
        return []

def parse_html_vacancies(html, city_id):
    vacancies = []
    vacancy_blocks = re.findall(r'data-qa="vacancy-serp__vacancy-title"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', html)
    for link, title in vacancy_blocks[:20]:
        vacancy = {}
        match = re.search(r"/vacancy/(\d+)", link)
        vacancy["id"] = match.group(1) if match else link
        vacancy["name"] = title.strip()
        vacancy["alternate_url"] = link if link.startswith("http") else f"https://hh.ru{link}"
        company_match = re.search(r'data-qa="vacancy-serp__vacancy-employer"[^>]*>([^<]+)</a>', html)
        vacancy["employer"] = {"name": company_match.group(1).strip() if company_match else "Не указана"}
        city_match = re.search(r'data-qa="vacancy-serp__vacancy-address"[^>]*>([^<]+)</span>', html)
        vacancy["area"] = {"name": city_match.group(1).strip() if city_match else city_id}
        salary_match = re.search(r'data-qa="vacancy-serp__vacancy-compensation"[^>]*>([^<]+)</span>', html)
        vacancy["salary"] = parse_salary(salary_match.group(1)) if salary_match else None
        desc_match = re.search(r'data-qa="vacancy-serp__vacancy_snippet_requirement"[^>]*>([^<]+)</span>', html)
        desc = desc_match.group(1) if desc_match else ""
        vacancy["description"] = desc
        vacancy["snippet"] = {"requirement": desc[:300] + "..." if len(desc) > 300 else desc, "responsibility": ""}
        vacancies.append(vacancy)
    logger.info(f"HTML fallback: распарсено {len(vacancies)} вакансий для {city_id}")
    return vacancies

def parse_salary(text):
    if not text:
        return None
    clean_text = text.replace("\xa0", " ").replace("\u00a0", " ").replace("\u202f", " ")
    clean_text = clean_text.replace(" ", "").replace("\t", "").replace("\n", "")
    match = re.search(r"(от)?(\d+)([-—](\d+))?(₽|руб|RUB|rub)", clean_text, re.IGNORECASE)
    if not match:
        return None
    try:
        salary_from = int(match.group(2)) if match.group(2) else None
        salary_to = int(match.group(4)) if match.group(4) else None
        return {"from": salary_from, "to": salary_to, "currency": "RUR"}
    except (ValueError, TypeError):
        return None

def parse_rss(xml_text):
    vacancies = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            vacancy = {}
            link = item.findtext("link", "")
            match = re.search(r"/vacancy/(\d+)", link)
            vacancy["id"] = match.group(1) if match else link
            vacancy["name"] = item.findtext("title", "Без названия")
            vacancy["alternate_url"] = link
            vacancy["published"] = item.findtext("pubDate", "")
            desc = item.findtext("description", "")
            vacancy["description"] = desc
            title = vacancy["name"]
            if " — " in title:
                parts = title.rsplit(" — ", 1)
                vacancy["name"] = parts[0].strip()
                vacancy["employer"] = {"name": parts[1].strip()}
            else:
                vacancy["employer"] = {"name": "Не указана"}
            city_match = re.search(r"([А-Яа-я\s-]+)(?:,|\s*•|\s*—)", desc)
            vacancy["area"] = {"name": city_match.group(1).strip() if city_match else "Не указан"}
            vacancy["salary"] = parse_salary(desc)
            vacancy["snippet"] = {"requirement": desc[:300] + "..." if len(desc) > 300 else desc, "responsibility": ""}
            vacancies.append(vacancy)
    except Exception as e:
        logger.error(f"Ошибка обработки RSS: {e}")
    return vacancies

def is_relevant_by_keywords(vacancy):
    text = (vacancy.get("name", "") + " " + vacancy.get("description", "") + 
            " " + vacancy.get("snippet", {}).get("requirement", "")).lower()
    if not any(kw.lower() in text for kw in USER_FILTERS["industry_keywords"]):
        return False
    if any(ew.lower() in text for ew in USER_FILTERS["exclude_words"]):
        return False
    return True

async def ask_deepseek(vacancy):
    global deepseek_available, client
    if not deepseek_available and DEEPSEEK_API_KEY:
        try:
            http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
            client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", http_client=http_client)
            test = await client.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": "Привет"}], 
                max_tokens=5
            )
            deepseek_available = True
            logger.info("DeepSeek восстановлен")
        except Exception as e:
            logger.debug(f"DeepSeek всё ещё недоступен: {e}")
            client = None
    if not deepseek_available or not client:
        logger.info("DeepSeek отключен, пропускаю AI-фильтрацию")
        return True

    prompt = f"""Ты — профессиональный рекрутер-аналитик с 20-летним опытом в подборе топ-менеджеров в нефтяной отрасли.
Твоя задача — оценить, подходит ли вакансия кандидату со следующим профилем:

{CANDIDATE_PROFILE}

=== ВАКАНСИЯ ===
Название: {vacancy.get('name', '')}
Компания: {vacancy.get('employer', {}).get('name', '')}
Город: {vacancy.get('area', {}).get('name', '')}
Зарплата: {vacancy.get('salary', {}).get('from', '')} - {vacancy.get('salary', {}).get('to', '')} {vacancy.get('salary', {}).get('currency', '')}
Требования: {vacancy.get('snippet', {}).get('requirement', '')}
Обязанности: {vacancy.get('snippet', {}).get('responsibility', '')}
Полное описание: {vacancy.get('description', '')[:2000]}

=== КРИТЕРИИ ОЦЕНКИ ===
1. Индустрия — обязательно нефтепродукты, ГСМ, топливо, нефтебаза, АЗС, СПбМТСБ, НПЗ.
2. Должность — коммерческий директор, руководитель отдела продаж, директор по продажам, директор филиала.
3. Зарплата — от 200 000 ₽ на руки.
4. Город — Волгоград, Москва, Казань, НН, Уфа, Астана, Баку, Минск.
5. Занятость — полная, формат офис/производство.
6. Опыт — более 6 лет, B2B-продажи, контракты с НПЗ.

Ответь строго "ДА" или "НЕТ". Если сомневаешься, но кандидат может быть полезен — ответь "ДА"."""

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat", 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.1, 
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("ДА")
    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        deepseek_available = False
        client = None
        return True

def format_vacancy_message(vacancy):
    name = vacancy.get('name', 'Без названия')
    company = vacancy.get('employer', {}).get('name', 'Не указана')
    city = vacancy.get('area', {}).get('name', 'Не указан')
    salary = vacancy.get('salary')
    if salary:
        salary_text = f"{salary.get('from', '')} - {salary.get('to', '')} {salary.get('currency', '')}".strip().replace("None", "").strip() or "Не указана"
    else:
        salary_text = "Не указана"
    desc = vacancy.get('description', '')
    desc_clean = re.sub('<[^<]+?>', '', desc)
    desc_short = desc_clean[:400] + "..." if len(desc_clean) > 400 else desc_clean
    url = vacancy.get('alternate_url', '')
    return f"""━━━━━━━━━━━━━━━━━━━━
🔹 {name}
🏢 {company}
📍 {city}
💰 {salary_text}

📋 Описание:
{desc_short}

🔗 {url}

✅ Почему подходит Виктору:
Опыт в управлении продажами ГСМ, закупках на СПбМТСБ и логистике нефтепродуктов.

💡 Совет:
В сопроводительном письме подчеркните опыт работы с СПбМТСБ и управления нефтебазой.
━━━━━━━━━━━━━━━━━━━━"""

async def background_search(context: ContextTypes.DEFAULT_TYPE):
    global deepseek_available
    logger.info("🔄 Запуск фонового поиска...")
    chat_id = None
    if context.job and hasattr(context.job, 'chat_id') and context.job.chat_id:
        chat_id = context.job.chat_id
    elif CHAT_ID:
        chat_id = int(CHAT_ID)
    if not chat_id:
        logger.error("Нет chat_id для отправки")
        return

    all_vacancies = []
    async with aiohttp.ClientSession() as session:
        for city_ru, city_id in USER_FILTERS["cities"].items():
            city_vacancies = []
            for keyword in USER_FILTERS["keywords"]:
                logger.info(f"Поиск: {keyword} в {city_ru}")
                result = await fetch_rss(session, city_id, keyword, per_page=20)
                if result:
                    city_vacancies.extend(result)
                else:
                    fallback = await fetch_html_fallback(session, city_id, keyword)
                    city_vacancies.extend(fallback)
                await asyncio.sleep(2 + random.uniform(0, 2))
            all_vacancies.extend(city_vacancies)
            logger.info(f"Город {city_ru}: {len(city_vacancies)} вакансий")
            await asyncio.sleep(3 + random.uniform(1, 3))

    seen = set()
    unique = [v for v in all_vacancies if not (v['id'] in seen or seen.add(v['id']))]
    new_vacancies = [v for v in unique if not is_vacancy_sent(v['id'])]
    logger.info(f"Найдено {len(unique)} уникальных, новых: {len(new_vacancies)}")

    if not new_vacancies:
        await context.bot.send_message(chat_id=chat_id, text="🔍 Новых вакансий не найдено.")
        log_search(0, 0)
        return

    keyword_filtered = [v for v in new_vacancies if is_relevant_by_keywords(v)]
    logger.info(f"После keyword-фильтра: {len(keyword_filtered)}")

    matched = []
    for v in keyword_filtered:
        if await ask_deepseek(v):
            matched.append(v)
            mark_vacancy_sent(v['id'], v.get('name', ''), v.get('employer', {}).get('name', ''))
        await asyncio.sleep(0.5)

    logger.info(f"После фильтрации: {len(matched)}")
    log_search(len(unique), len(matched))
    ds_status = "✅ с AI-фильтром" if deepseek_available else "⚠️ без AI"

    if matched:
        await context.bot.send_message(chat_id=chat_id, text=f"🔍 Найдено {len(matched)} вакансий {ds_status}:")
        for v in matched:
            await context.bot.send_message(chat_id=chat_id, text=format_vacancy_message(v))
            await asyncio.sleep(2)
    else:
        await context.bot.send_message(chat_id=chat_id, text=f"🔍 Подходящих вакансий не найдено {ds_status}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global deepseek_available
    if not deepseek_available and DEEPSEEK_API_KEY:
        try:
            http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
            test_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1", http_client=http_client)
            test = await test_client.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": "Привет"}], 
                max_tokens=5
            )
            deepseek_available = True
        except:
            pass
    ds_status = "✅ подключен" if deepseek_available else "❌ отключен"
    scraper_status = "✅ ScraperAPI" if SCRAPERAPI_KEY else "❌ прокси"
    await update.message.reply_text(
        f"👋 Привет! Я ищу вакансии коммерческого директора в нефтянке.\n\n"
        f"🤖 DeepSeek: {ds_status}\n🌐 {scraper_status}\n📡 HH.ru (RSS + HTML)\n\n"
        "Команды:\n/search — поиск сейчас\n/schedule — авто on/off\n/stats — статистика\n/filters — фильтры\n/salary [сумма]\n/relocate — города\n/cleanup — очистить\n/help — помощь"
    )

async def search_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text("🔍 Ищу вакансии, подождите...")
    try:
        class JobProxy:
            def __init__(self, cid):
                self.chat_id = cid
        original_job = getattr(context, 'job', None)
        context.job = JobProxy(chat_id)
        await background_search(context)
        context.job = original_job
        await update.message.reply_text("✅ Поиск завершён!")
    except Exception as e:
        logger.error(f"Ошибка в search_now: {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)[:200]}")

async def schedule_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        jobs = context.job_queue.get_jobs_by_name("auto_search")
        if jobs:
            await update.message.reply_text("⏰ Авто-поиск активен (9:00 и 18:00 UTC)\nОтключить: /schedule off")
        else:
            await update.message.reply_text("❌ Авто-поиск отключен\nВключить: /schedule on")
        return
    command = context.args[0].lower()
    for job in context.job_queue.get_jobs_by_name("auto_search"):
        job.schedule_removal()
    if command == "off":
        await update.message.reply_text("❌ Авто-поиск отключён")
        return
    if command in ["on", "twice"]:
        context.job_queue.run_daily(background_search, time=datetime.strptime("09:00", "%H:%M").time(), chat_id=chat_id, name="auto_search")
        context.job_queue.run_daily(background_search, time=datetime.strptime("18:00", "%H:%M").time(), chat_id=chat_id, name="auto_search")
        await update.message.reply_text("✅ Авто-поиск включён!\n• 9:00 UTC\n• 18:00 UTC\nОтключить: /schedule off")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        total_sent, total_searches, recent = get_stats()
        text = f"📊 Статистика:\n\n• Всего отправлено: {total_sent}\n• Всего поисков: {total_searches}\n\n"
        if recent:
            text += "Последние поиски:\n"
            for found, sent, when in recent:
                text += f"  {str(when)[:16]} — найдено {found}, подошло {sent}\n"
        await update.message.reply_text(text)
    except Exception as e:
        logger.error(f"Ошибка stats: {e}")
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)[:200]}")

async def filters_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cities_str = ", ".join(USER_FILTERS["cities"].keys())
    await update.message.reply_text(f"🔧 Фильтры:\n• Мин. зарплата: {USER_FILTERS['salary_min']} ₽\n• Города: {cities_str}")

async def set_salary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Укажите сумму: /salary 250000")
        return
    try:
        new_salary = int(context.args[0])
        if new_salary < 100000:
            await update.message.reply_text("Минимум 100 000 ₽")
            return
        USER_FILTERS["salary_min"] = new_salary
        await update.message.reply_text(f"✅ Мин. зарплата: {new_salary} ₽")
    except ValueError:
        await update.message.reply_text("Введите число")

async def relocate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🌍 Города: Волгоград, Астана, Баку, Казань, Минск, Нижний Новгород, Уфа")

async def cleanup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    deleted = cleanup_old_vacancies(days=30)
    await update.message.reply_text(f"🗑 Удалено {deleted} старых записей")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""📖 Команды:
/search — поиск сейчас
/schedule on/off — авто-поиск
/stats — статистика
/filters — фильтры
/salary 250000 — зарплата
/relocate — города
/cleanup — очистить
/help — справка""")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Ошибка: {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text("⚠️ Ошибка. Попробуйте позже.")

# ========== WEB SERVER + WEBHOOK ==========
app = FastAPI()

@app.get("/")
async def root():
    return {"status": "alive", "bot": "hh-bot", "deepseek": deepseek_available, "time": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)  # <-- await добавлен
    return {"ok": True}

application = None

async def keep_alive_ping():
    while True:
        await asyncio.sleep(300)
        logger.info("💓 Keep-alive ping")

async def run_webhook():
    global application
    init_db()

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("search", search_now))
    application.add_handler(CommandHandler("schedule", schedule_search))
    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(CommandHandler("filters", filters_cmd))
    application.add_handler(CommandHandler("salary", set_salary))
    application.add_handler(CommandHandler("relocate", relocate))
    application.add_handler(CommandHandler("cleanup", cleanup_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_error_handler(error_handler)

    # ВСЕ await добавлены
    await application.initialize()
    await application.start()

    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await application.bot.set_webhook(url=webhook_url)  # <-- await
        logger.info(f"🔗 Webhook установлен: {webhook_url}")

    # Запускаем keep-alive как фоновую задачу
    asyncio.create_task(keep_alive_ping())

    # Запускаем uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=10000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(run_webhook())
