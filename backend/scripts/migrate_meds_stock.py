from config import engine
from sqlalchemy import text

def migrate():
    with engine.connect() as conn:
        print("Migrating medications table...")
        # Add current_stock
        try:
            conn.execute(text("ALTER TABLE medications ADD COLUMN IF NOT EXISTS current_stock INTEGER DEFAULT 0"))
            print("  OK: current_stock added")
        except Exception as e:
            print(f"  Error adding current_stock: {e}")

        # Add doses_per_day
        try:
            conn.execute(text("ALTER TABLE medications ADD COLUMN IF NOT EXISTS doses_per_day INTEGER DEFAULT 1"))
            print("  OK: doses_per_day added")
        except Exception as e:
            print(f"  Error adding doses_per_day: {e}")
        
        conn.commit()
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
