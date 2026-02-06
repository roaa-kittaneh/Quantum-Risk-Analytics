import pandas as pd
import numpy as np
import os

def prepare_crisis_data():
    print("--- Phase 1: Large-Scale Crisis Data Preparation ---")
    
    # 1. توليد مدى زمني واسع (34 سنة من البيانات التاريخية)
    # هذا يعطينا حوالي 8,800+ يوم تداول، مما يثبت قدرة المشروع على التعامل مع البيانات الضخمة
    dates = pd.date_range(start="1990-01-01", end="2024-01-01", freq='B')
    n_days = len(dates)
    print(f"Processing {n_days} days of historical market records...")
    
    np.random.seed(42) # لضمان ثبات النتائج في كل مرة
    
    # 2. استخدام توزيع Student-t (Fat Tails) لنمذجة الأزمات
    # هذا التوزيع هو الذي يسمح بوجود "أحداث متطرفة" (Black Swans)
    gold_returns = np.random.standard_t(df=3, size=n_days) * 0.01 
    sp500_returns = np.random.standard_t(df=3, size=n_days) * 0.012
    
    # 3. إضافة "صدمات عالمية" (Global Event Shocks)
    # نقوم بمحاكاة 5 أزمات كبرى مفاجئة (مثل أزمة 2008 أو كورونا)
    for _ in range(5):
        shock_index = np.random.randint(0, n_days)
        gold_returns[shock_index] -= 0.05  # هبوط مفاجئ 5%
        sp500_returns[shock_index] -= 0.08 # هبوط مفاجئ 8%
    
    # حفظ بيانات السوق
    returns_df = pd.DataFrame({'Gold': gold_returns, 'SP500': sp500_returns}, index=dates)
    returns_df.to_csv('market_returns.csv')
    print("SUCCESS: Crisis-ready market data with 'Fat Tails' saved to market_returns.csv")

    # 4. توليد بيانات قروض ضخمة (100,000 سجل)
    # هذا يثبت قدرة النظام على تجميع المخاطر (Risk Aggregation) لمحفظة كبيرة
    n_loans = 100000
    print(f"Generating large-scale credit portfolio: {n_loans} loan records...")
    
    loan_data = pd.DataFrame({
        'LoanID': range(n_loans),
        'Default': np.random.choice([0, 1], size=n_loans, p=[0.92, 0.08]), # نسبة تعثر 8% (عالية بسبب الأزمة)
        'Amount': np.random.normal(15000, 5000, n_loans)
    })
    loan_data.to_csv('loan_data.csv', index=False)
    print(f"SUCCESS: {n_loans} loan records saved to loan_data.csv")
    
    print("\n--- Data Preparation Complete! ---")
    print("You can now run 'flask_api_server.py' to start the Quantum Backend.")

if __name__ == "__main__":
    prepare_crisis_data()
