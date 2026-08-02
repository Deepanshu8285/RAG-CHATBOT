import sqlite3
import os

def get_db_path(username):
    user_dir = f"user_data/{username}"
    os.makedirs(user_dir, exist_ok=True)
    return f"{user_dir}/chat_history.db"

def init_db(username):
    conn = sqlite3.connect(get_db_path(username))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_message(username, question, answer):
    conn = sqlite3.connect(get_db_path(username))
    conn.execute("INSERT INTO chat_history (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

def load_history(username):
    conn = sqlite3.connect(get_db_path(username))
    rows = conn.execute("SELECT question, answer FROM chat_history ORDER BY id ASC").fetchall()
    conn.close()
    return rows

def clear_history(username):
    conn = sqlite3.connect(get_db_path(username))
    conn.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()