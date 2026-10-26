"""路径与常量。所有模块从这里取路径,不各自拼。"""
from __future__ import annotations
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "ashare.db")
CONTRACT_PATH = os.path.join(ROOT, "contracts", "ashare.yaml")
SCHEMA_PATH = os.path.join(ROOT, "ashare_forge", "data_schema.sql")
