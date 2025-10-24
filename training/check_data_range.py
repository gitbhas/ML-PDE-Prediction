#!/usr/bin/env python3
"""
Check date range in training data
Usage: python check_data_range.py
"""
import json
import pandas as pd
from datetime import datetime

def check_file_audit_dates():
    """Check date range in file_audit.json"""
    dates = []
    
    try:
        with open('../lambda/file_audit.json', 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    date_str = record.get('fil_creatn_dt', '')[:10]
                    if date_str:
                        dates.append(date_str)
                except:
                    continue
        
        if dates:
            dates.sort()
            print(f"File Audit Data Range:")
            print(f"  Start Date: {dates[0]}")
            print(f"  End Date: {dates[-1]}")
            print(f"  Total Records: {len(dates)}")
            print(f"  Unique Dates: {len(set(dates))}")
        else:
            print("No valid dates found in file_audit.json")
            
    except Exception as e:
        print(f"Error reading file_audit.json: {e}")

def check_runtime_dates():
    """Check date range in PDE-Runtimes-2025.csv"""
    try:
        df = pd.read_csv('../lambda/PDE-Runtimes-2025.csv')
        
        print(f"\nRuntime CSV Columns: {list(df.columns)}")
        
        # Try different date parsing approaches
        try:
            dates = pd.to_datetime(df['PDE date'], format='%d %B %Y').dt.strftime('%Y-%m-%d')
        except:
            try:
                dates = pd.to_datetime(df['PDE date'], dayfirst=True).dt.strftime('%Y-%m-%d')
            except:
                dates = pd.to_datetime(df['PDE date'], format='mixed').dt.strftime('%Y-%m-%d')
        
        print(f"\nRuntime Data Range:")
        print(f"  Start Date: {dates.min()}")
        print(f"  End Date: {dates.max()}")
        print(f"  Total Records: {len(dates)}")
        print(f"  Unique Dates: {len(dates.unique())}")
        print(f"  Sample data:")
        print(df.head())
        
    except Exception as e:
        print(f"Error reading PDE-Runtimes-2025.csv: {e}")

if __name__ == "__main__":
    check_file_audit_dates()
    check_runtime_dates()