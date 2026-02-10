import pandas as pd
from datetime import datetime
import db_config

if db_config.USE_SUPABASE:
    from supabase import create_client
    supabase_client = create_client(db_config.SUPABASE_URL, db_config.SUPABASE_KEY)
else:
    import sqlite3

class DatabaseManager:
    """Unified database interface that works with both SQLite and Supabase"""
    
    def __init__(self):
        self.use_supabase = db_config.USE_SUPABASE
        self.dbname = db_config.DBNAME if db_config.USE_SQLITE else None
        
    def read_table(self, table_name, filters=None, columns='*', order_by=None, limit=None):
        """Read data from a table
        
        Args:
            table_name: Name of the table
            filters: Dict of column:value pairs for WHERE conditions
            columns: Comma-separated column names or '*'
            order_by: Column name to sort by (add '-' prefix for DESC)
            limit: Max number of rows to return
        """
        if self.use_supabase:
            query = supabase_client.table(table_name).select(columns)
            
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            
            if order_by:
                # Convert SQLite syntax to Supabase syntax
                # SQLite: 'column' or '-column' or 'col1, -col2'
                # Supabase: order('column', desc=False) or order('column', desc=True)
                orders = [o.strip() for o in order_by.split(',')]
                for order_col in orders:
                    if order_col.startswith('-'):
                        # Descending
                        col_name = order_col[1:]
                        query = query.order(col_name, desc=True)
                    else:
                        # Ascending
                        query = query.order(order_col, desc=False)
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()  # FIXED: Execute the query
            return pd.DataFrame(response.data)  # FIXED: Return the dataframe
        else:
            # SQLite
            with sqlite3.connect(self.dbname) as conn:
                where_clause = ""
                params = []
                
                if filters:
                    conditions = [f"{k} = ?" for k in filters.keys()]
                    where_clause = f" WHERE {' AND '.join(conditions)}"
                    params = list(filters.values())
                
                order_clause = ""
                if order_by:
                    if order_by.startswith('-'):
                        order_clause = f' ORDER BY "{order_by[1:]}" DESC'
                    else:
                        order_clause = f' ORDER BY "{order_by}"'
                
                limit_clause = f" LIMIT {limit}" if limit else ""
                
                query = f'SELECT {columns} FROM "{table_name}"{where_clause}{order_clause}{limit_clause}'

                return pd.read_sql_query(query, conn, params=params if params else None)
    
    def insert_records(self, table_name, records, upsert=False):
        """Insert records into a table
        
        Args:
            table_name: Name of the table
            records: List of dicts or single dict
            upsert: If True, update on conflict (Supabase only)
        """
        if isinstance(records, dict):
            records = [records]

        # Normalize Date Fields to YYYY-MM-DD format
        for record in records:
            if 'Date' in record:
                record['Date'] = pd.to_datetime(record['Date'], format='mixed', errors='coerce').strftime('%Y-%m-%d')
        
        if self.use_supabase:
            # Batch insert in chunks of 1000 to avoid Supabase limits
            batch_size = 1000
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                if upsert:
                    supabase_client.table(table_name).upsert(batch).execute()
                else:
                    supabase_client.table(table_name).insert(batch).execute()
        else:
            # SQLite
            with sqlite3.connect(self.dbname) as conn:
                if upsert:
                    # Use INSERT OR REPLACE for upsert to handle UNIQUE constraints
                    df = pd.DataFrame(records)
                    if df.empty:
                        return

                    columns = ', '.join([f'"{col}"' for col in df.columns])
                    placeholders = ', '.join(['?'] * len(df.columns))
                    sql = f'INSERT OR REPLACE INTO "{table_name}" ({columns}) VALUES ({placeholders})'

                    for _, row in df.iterrows():
                        conn.execute(sql, tuple(row))
                    conn.commit()
                else:
                    # Regular insert
                    df = pd.DataFrame(records)
                    df.to_sql(table_name, conn, if_exists='append', index=False)
    
    def delete_records(self, table_name, filters):
        """Delete records from a table
        
        Args:
            table_name: Name of the table
            filters: Dict of column:value pairs for WHERE conditions
        """
        if self.use_supabase:
            query = supabase_client.table(table_name).delete()
            for key, value in filters.items():
                query = query.eq(key, value)
            return query.execute()
        else:
            # SQLite
            with sqlite3.connect(self.dbname) as conn:
                conditions = [f"{k} = ?" for k in filters.keys()]
                where_clause = f" WHERE {' AND '.join(conditions)}"
                params = list(filters.values())
                
                conn.execute(f'DELETE FROM "{table_name}"{where_clause}', params)

                conn.commit()
    
    def delete_all_records(self, table_name):
        """Delete all records from a table"""
        if self.use_supabase:
            try:
                print(f"   Fetching records from {table_name}...")
                response = supabase_client.table(table_name).select('*').limit(10000).execute()
                
                print(f"   Found {len(response.data)} records")
                
                if not response.data:
                    print(f"   No data found!")
                    return
                
                first_record = response.data[0]
                print(f"   First record keys: {list(first_record.keys())}")
                
                # Determine primary key
                if 'Date' in first_record:
                    pk_column = 'Date'
                elif 'id' in first_record:
                    pk_column = 'id'
                elif 'tscode' in first_record:
                    pk_column = 'tscode'
                elif 'ticker' in first_record:
                    pk_column = 'ticker'
                elif 'scan_date' in first_record:
                    pk_column = 'scan_date'
                else:
                    print(f"   ❌ Cannot find primary key. Available keys: {list(first_record.keys())}")
                    return
                
                print(f"   Using primary key: {pk_column}")
                print(f"   Deleting {len(response.data)} records...")
                
                # Delete one by one with confirmation
                deleted_count = 0
                for i, record in enumerate(response.data):
                    try:
                        delete_response = supabase_client.table(table_name).delete().eq(pk_column, record[pk_column]).execute()
                        deleted_count += 1
                        if (i + 1) % 100 == 0:
                            print(f"   Deleted {i + 1}/{len(response.data)}...")
                    except Exception as e:
                        print(f"   ❌ Failed to delete record {i}: {e}")
                
                print(f"   ✅ Deleted {deleted_count} records")
                
            except Exception as e:
                print(f"   ❌ Error in delete_all_records: {e}")
                import traceback
                traceback.print_exc()
        else:
            with sqlite3.connect(self.dbname) as conn:
                conn.execute(f'DELETE FROM "{table_name}"')
                conn.commit()

    
    def execute_raw_sql(self, query, params=None):
        """Execute raw SQL (SQLite only)"""
        if self.use_supabase:
            raise NotImplementedError("Raw SQL not supported for Supabase, use specific methods")
        else:
            with sqlite3.connect(self.dbname) as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return cursor
    
    def table_exists(self, table_name):
        """Check if a table exists"""
        if self.use_supabase:
            try:
                # Try to select from table
                supabase_client.table(table_name).select('*').limit(1).execute()
                return True
            except:
                return False
        else:
            # SQLite
            with sqlite3.connect(self.dbname) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                return cursor.fetchone() is not None
    
    def create_table_sqlite(self, schema):
        """Create a table with given schema (SQLite only)"""
        if self.use_supabase:
            print("⚠️  Tables must be created in Supabase dashboard first")
            return False
        else:
            with sqlite3.connect(self.dbname) as conn:
                conn.execute(schema)
                conn.commit()
            return True
    
    def get_connection(self):
        """Get raw connection (SQLite only)"""
        if self.use_supabase:
            raise NotImplementedError("Raw connection not supported for Supabase")
        else:
            return sqlite3.connect(self.dbname)

# Global instance
db = DatabaseManager()
