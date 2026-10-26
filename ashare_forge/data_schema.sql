-- AShare Forge SQLite schema v1 (data/ashare.db)
CREATE TABLE IF NOT EXISTS stocks (
  code TEXT PRIMARY KEY,          -- 'sz.000001'
  name TEXT,
  board TEXT,                     -- main|chinext|star|bse
  first_seen TEXT,                -- 首次出现在快照/或K线首日 YYYY-MM-DD
  last_seen TEXT,                 -- 最后出现日
  delisted INTEGER DEFAULT 0,
  industry TEXT, industry_src TEXT, industry_asof TEXT
);
CREATE TABLE IF NOT EXISTS daily (
  code TEXT NOT NULL, date TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, preclose REAL,
  volume REAL, amount REAL, turn REAL,
  tradestatus INTEGER, pctChg REAL,
  peTTM REAL, pbMRQ REAL, psTTM REAL, pcfNcfTTM REAL, isST INTEGER,
  PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily(date);
CREATE TABLE IF NOT EXISTS adj_factor (
  code TEXT NOT NULL, date TEXT NOT NULL,   -- 因子生效日(除权日)
  foreAdjustFactor REAL, backAdjustFactor REAL, adjustFactor REAL,
  PRIMARY KEY (code, date)
);
CREATE TABLE IF NOT EXISTS index_daily (
  code TEXT NOT NULL, date TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume REAL, amount REAL, pctChg REAL,
  PRIMARY KEY (code, date)
);
CREATE TABLE IF NOT EXISTS universe_snapshot (
  day TEXT NOT NULL, code TEXT NOT NULL, tradestatus INTEGER, name TEXT,
  PRIMARY KEY (day, code)
);
CREATE TABLE IF NOT EXISTS sync_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, task TEXT, code TEXT,
  rows INTEGER, status TEXT, message TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
