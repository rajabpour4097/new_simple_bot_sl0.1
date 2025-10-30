# Commission Coverage Feature

## 📋 Overview
این ویژگی به صورت خودکار Stop Loss را به نقطه‌ای بعد از کمیسیون منتقل می‌کند تا از ضرر کمیسیون در صورت برگشت قیمت جلوگیری شود.

## ⚙️ How It Works

### 1. محاسبه خودکار نقطه کمیسیون
وقتی یک پوزیشن باز می‌شود:
- **ریسک پولی (1R)** محاسبه می‌شود: `risk_money = risk_pips × pip_value × volume`
- **کمیسیون به R** تبدیل می‌شود: `commission_R = commission_per_lot / risk_money`
- **بافر اضافی** اضافه می‌شود: `final_trigger = commission_R + buffer_R`

### 2. انتقال خودکار SL
وقتی سود به `commission_R + buffer` می‌رسد:
- **SL** به همان نقطه trigger منتقل می‌شود
- **TP** بدون تغییر باقی می‌ماند (همان 2R اولیه)
- این عمل **فقط یک بار** انجام می‌شود

### 3. مراحل بعدی
پس از پوشش کمیسیون، مراحل 2R، 3R، 4R و ... به همان صورت قبلی ادامه می‌یابند.

## 📊 Configuration

در `metatrader5_config.py`:

```python
DYNAMIC_RISK_CONFIG = {
    'commission_per_lot': 4.5,  # کمیسیون کل (دلار) - طبق بروکر تنظیم کنید
    'commission_coverage_stage': {
        'enable': True,           # فعال/غیرفعال
        'commission_buffer_R': 0.15,  # بافر اضافی (15% از 1R)
        'auto_calculate': True,   # محاسبه خودکار
    },
    'stages': [
        {
            'id': 'stage_commission_coverage',
            'trigger_R': 'auto',  # محاسبه خودکار
            'sl_lock_R': 'auto',  # قفل روی همان trigger
            'tp_R': None          # TP بدون تغییر
        },
        # مراحل 2R به بعد...
    ]
}
```

## 📈 Example Scenario

فرض کنید:
- **Risk**: 1% از balance
- **Risk Money**: $10
- **Commission**: $4.5

محاسبه:
```
commission_R = 4.5 / 10 = 0.45R
buffer = 0.15R
trigger = 0.45 + 0.15 = 0.60R
```

جریان کار:
1. ✅ پوزیشن باز می‌شود با SL اصلی و TP=2R
2. 📊 قیمت به 0.60R می‌رسد
3. 💰 SL به 0.60R منتقل می‌شود (TP همچنان 2R)
4. 🎯 قیمت به 2R می‌رسد
5. ⚙️ SL به 2R منتقل می‌شود، TP به 3R تغییر می‌کند
6. 🔄 و همینطور ادامه...

## 🎯 Benefits

✅ **محافظت از سرمایه**: حداقل کمیسیون پوشش داده می‌شود  
✅ **خودکار**: نیازی به محاسبه دستی نیست  
✅ **انعطاف‌پذیر**: buffer قابل تنظیم  
✅ **بدون تداخل**: TP اصلی تغییر نمی‌کند  

## ⚠️ Important Notes

- این ویژگی **قبل از مرحله 2R** فعال می‌شود
- محاسبات بر اساس **volume واقعی** پوزیشن است
- اگر محاسبه ناموفق باشد، به `0.1R` برمی‌گردد
- لاگ‌ها جزئیات کامل محاسبات را نمایش می‌دهند

## 🔧 Troubleshooting

اگر commission coverage فعال نشد:
1. بررسی کنید `enable: True` باشد
2. بررسی کنید `commission_per_lot` صحیح باشد
3. لاگ‌ها را برای پیام `Commission calc:` بررسی کنید
4. اگر لازم است `commission_buffer_R` را تنظیم کنید
