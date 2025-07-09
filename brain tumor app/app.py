import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
import flask
import requests

# --- تنظیمات ---
GROQ_API_KEY = ""
MODEL_PATH = "model_full.pt"
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

SYMPTOM_DETAILS = {
    "سردرد": "سردرد می‌تواند نشانه افزایش فشار داخل جمجمه یا تومور باشد.",
    "تشنج": "تشنج اغلب در تومورهای لوب فرونتال یا تمپورال دیده می‌شود.",
    "اختلال در بینایی": "اختلال بینایی می‌تواند ناشی از فشار تومور بر مسیرهای بینایی باشد.",
    "اختلال در تعادل": "اختلال تعادل معمولاً در تومورهای مخچه یا ساقه مغز رخ می‌دهد.",
    "سرگیجه": "سرگیجه می‌تواند نشانه اختلال در گردش خون مغزی یا فشار تومور باشد.",
    "دوبینی": "دوبینی معمولاً به علت فشار بر اعصاب حرکتی چشم ایجاد می‌شود.",
    "بی‌حسی اندام": "بی‌حسی اندام‌ها می‌تواند ناشی از درگیری قشر حرکتی یا حسی مغز باشد.",
    "اختلال گفتار": "اختلال گفتار در تومورهای نیمکره غالب یا ساقه مغز دیده می‌شود.",
    "کاهش شنوایی": "کاهش شنوایی در تومورهای زاویه مخچه یا ساقه مغز شایع است.",
    "کاهش وزن": "کاهش وزن می‌تواند نشانه بیماری مزمن یا بدخیمی باشد.",
    "خستگی": "خستگی علامت غیر اختصاصی اما شایع در بیماری‌های مزمن است.",
    "افسردگی": "افسردگی می‌تواند ناشی از اثرات روانی یا مستقیم تومور باشد.",
    "اضطراب": "اضطراب در بیماران با بیماری‌های مزمن شایع است.",
    "تاری دید": "تاری دید می‌تواند ناشی از ادم پاپی یا فشار بر عصب بینایی باشد.",
    "بی‌اختیاری ادرار": "بی‌اختیاری ادرار در تومورهای لوب فرونتال یا نخاعی دیده می‌شود.",
    "اختلال بلع": "اختلال بلع معمولاً در تومورهای ساقه مغز رخ می‌دهد.",
    "درد گردن": "درد گردن می‌تواند نشانه درگیری مننژ یا نخاع باشد.",
    "خواب‌آلودگی غیرعادی": "خواب‌آلودگی می‌تواند نشانه افزایش فشار داخل جمجمه باشد.",
    "کاهش تمرکز": "کاهش تمرکز در تومورهای لوب فرونتال یا اثرات روانی شایع است.",
}

# --- Flask app ---
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- مدل و پیش‌پردازش ---
def load_model():
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        return CLASS_NAMES[pred.item()]

# --- API گروک ---
def get_medical_advice(symptoms, info):
    prompt = (
        f"Patient info: {info}\n"
        f"Symptoms (user selected and described): {symptoms}\n"
        "بر اساس علائم وارد شده توسط کاربر و نتیجه MRI، ابتدا یک جمع‌بندی کوتاه از مهم‌ترین علائم ارائه بده و علائم را تفسیر کن . سپس تشخیص افتراقی و تشخیص اصلی را با توجه به این علائم و تصویر MRI توضیح بده. برای هر علامت توضیح بده که چرا مهم است و چه نقشی در تشخیص دارد. سپس توصیه‌های پزشکی، آزمایش‌های پیشنهادی و درمان دارویی یا غیردارویی را دقیقاً متناسب با همین علائم و شرایط بیمار به زبان فارسی و با توضیح مختصر ارائه کن. اگر علائم هشدار وجود دارد، حتماً هشدار بده. پاسخ را فقط و فقط به زبان فارسی و با نگارش روان ارائه بده."
    )
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    if response.status_code == 200:
        try:
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"خطا در پردازش پاسخ GROQ: {str(e)}\nمتن پاسخ: {response.text}"
    else:
        return f"پاسخی از سرویس GROQ دریافت نشد. کد: {response.status_code}\nمتن پاسخ: {response.text}"

# --- توابع دانش‌بنیان پزشکی ---
def get_brain_tumor_diseases():
    brain_tumor_diseases = {
        "glioma": {
            "persian_name": "گلیوما",
            "description": "تومورهایی که از سلول‌های گلیال مغز منشأ می‌گیرند و شایع‌ترین نوع تومور اولیه مغز محسوب می‌شوند.",
            "subtypes": [
                "آستروسیتوما (Astrocytoma)",
                "گلیوبلاستوما (Glioblastoma - درجه IV)",
                "الیگودندروگلیوما (Oligodendroglioma)",
                "اپندیموما (Ependymoma)"
            ],
            "symptoms": [
                "سردرد مداوم و شدید",
                "تهوع و استفراغ",
                "تشنج",
                "ضعف یا فلج در اندام‌ها",
                "اختلال در بینایی",
                "تغییرات شخصیتی",
                "اختلال در تعادل",
                "مشکلات گفتاری"
            ],
            "type": "بدخیم",
            "risk_level": "بالا",
            "prognosis": "متغیر بسته به درجه و نوع",
            "common_locations": ["لوب فرونتال", "لوب پاریتال", "لوب تمپورال", "تنه مغز"],
            "age_group": "تمام سنین، اما بیشتر در بزرگسالان"
        },
        "meningioma": {
            "persian_name": "مننژیوما",
            "description": "تومورهایی که از مننژها (پرده‌های پوششی مغز و نخاع) منشأ می‌گیرند و معمولاً خوش‌خیم هستند.",
            "subtypes": [
                "مننژیوما خوش‌خیم (درجه I)",
                "مننژیوما آتیپیک (درجه II)",
                "مننژیوما بدخیم (درجه III)"
            ],
            "symptoms": [
                "سردرد",
                "تشنج",
                "ضعف تدریجی در اندام‌ها",
                "اختلال در بینایی",
                "تغییرات شخصیتی",
                "اختلال در حافظه",
                "مشکلات شنوایی",
                "گیجی"
            ],
            "type": "اکثراً خوش‌خیم",
            "risk_level": "متوسط",
            "prognosis": "معمولاً خوب",
            "common_locations": ["سطح مغز", "قاعده جمجمه", "کانال نخاعی"],
            "age_group": "بیشتر در زنان میانسال و سالمند"
        },
        "pituitary": {
            "persian_name": "آدنوم هیپوفیز",
            "description": "تومورهای خوش‌خیم غده هیپوفیز که می‌توانند هورمون تولید کنند یا غیرفعال باشند.",
            "subtypes": [
                "آدنوم‌های فعال هورمونی (Functioning)",
                "آدنوم‌های غیرفعال (Non-functioning)",
                "پرولاکتینوما",
                "آدنوم رشد (Growth hormone adenoma)",
                "بیماری کوشینگ (ACTH adenoma)"
            ],
            "symptoms": [
                "سردرد",
                "اختلال در بینایی (خاصه میدان دید)",
                "اختلالات هورمونی متنوع",
                "کاهش میل جنسی",
                "ناباروری",
                "تغییرات وزن",
                "خستگی مداوم",
                "افسردگی"
            ],
            "type": "خوش‌خیم",
            "risk_level": "کم تا متوسط",
            "prognosis": "معمولاً بسیار خوب",
            "common_locations": ["غده هیپوفیز در قاعده مغز"],
            "age_group": "تمام سنین، اما بیشتر در بزرگسالان"
        },
        "notumor": {
            "persian_name": "عدم وجود تومور",
            "description": "تصاویر طبیعی مغز بدون علائم تومور که ممکن است علائم مشابه به دلایل دیگری باشد.",
            "possible_causes": [
                "سردردهای میگرنی",
                "سردردهای تنشی",
                "اختلالات عروقی مغزی",
                "التهابات",
                "اختلالات روانی",
                "مشکلات هورمونی غیرتوموری",
                "فشار خون",
                "استرس و اضطراب"
            ],
            "symptoms": [
                "سردرد",
                "سرگیجه",
                "تهوع",
                "اختلال در تعادل",
                "تغییرات خلقی",
                "مشکلات حافظه",
                "خستگی",
                "اضطراب"
            ],
            "type": "غیرتوموری",
            "risk_level": "کم",
            "prognosis": "معمولاً خوب",
            "common_locations": ["مغز طبیعی"],
            "age_group": "تمام سنین"
        }
    }
    return brain_tumor_diseases

def get_diagnostic_features(tumor_type):
    """
    ویژگی‌های تشخیصی تصویری برای هر نوع تومور
    """
    features = {
        "glioma": [
            "توده با حدود نامشخص و نفوذی",
            "سیگنال ناهمگن در MRI",
            "ادم اطراف تومور",
            "تقویت کنندگی متغیر پس از تزریق گادولینیوم",
            "نکروز مرکزی (در درجات بالا)",
            "اثر فضاگیر (Mass effect)",
            "موقعیت داخل پارانشیم مغز",
            "تغییرات کیستیک احتمالی"
        ],
        "meningioma": [
            "توده با حدود مشخص",
            "اتصال به دورا ماتر (Dural tail)",
            "تقویت کنندگی یکنواخت",
            "سیگنال همگن",
            "موقعیت خارج محوری",
            "کلسیفیکاسیون احتمالی",
            "فشار روی پارانشیم مغز",
            "شکل کروی یا بیضی"
        ],
        "pituitary": [
            "توده در ناحیه سلار توسیکا",
            "تغییر شکل یا بزرگی غده هیپوفیز",
            "تقویت کنندگی متغیر",
            "فشار بر روی کیازم اپتیک",
            "گسترش به سینوس‌های اطراف (در موارد بزرگ)",
            "میکروآدنوم (کمتر از 1 سانتی‌متر)",
            "ماکروآدنوم (بیشتر از 1 سانتی‌متر)",
            "سیگنال متغیر بسته به نوع آدنوم"
        ],
        "notumor": [
            "ساختار طبیعی مغز",
            "عدم وجود توده یا ضایعه",
            "سیگنال طبیعی در MRI",
            "عدم تقویت کنندگی غیرطبیعی",
            "حفظ آناتومی طبیعی",
            "عدم ادم یا اثر فضاگیر",
            "بطن‌های طبیعی",
            "جریان خون طبیعی"
        ]
    }
    
    default_features = ["تصویر نیازمند بررسی دقیق‌تر توسط رادیولوژیست", "ارزیابی کلینیکی ضروری"]
    return features.get(tumor_type, default_features)

def get_treatments(tumor_type, patient_data):
    symptoms = patient_data.get('symptoms', [])
    treatments = []
    # توصیه اختصاصی برای هر علامت
    for symptom in symptoms:
        if symptom == "تشنج":
            treatments.append({"type": "دارویی", "treatment": "داروهای ضدتشنج مانند لوتیراستام یا فنی‌توئین", "priority": "بالا"})
        if symptom == "سردرد":
            treatments.append({"type": "دارویی", "treatment": "مسکن‌ها و در صورت نیاز کورتیکواستروئیدها برای کاهش ادم", "priority": "بالا"})
        if symptom == "اختلال در بینایی":
            treatments.append({"type": "تشخیصی", "treatment": "ارجاع فوری به پزشک مغز و اعصاب و بررسی ادم پاپی", "priority": "بالا"})
    if not treatments:
        base_treatments = {
            "glioma": [
                {"type": "جراحی", "treatment": "رزکسیون جراحی تا حد امکان (Maximal Safe Resection)", "priority": "بالا"},
                {"type": "رادیوتراپی", "treatment": "رادیوتراپی متعارف یا استریوتاکتیک", "priority": "بالا"},
                {"type": "شیمی‌درمانی", "treatment": "تمازولومید (Temozolomide)", "priority": "بالا"},
                {"type": "هدفمند", "treatment": "داروهای هدفمند براساس پروفایل مولکولی", "priority": "متوسط"},
                {"type": "حمایتی", "treatment": "کورتیکواستروئیدها برای کنترل ادم", "priority": "بالا"},
                {"type": "حمایتی", "treatment": "داروهای ضدتشنج", "priority": "بالا"},
                {"type": "توانبخشی", "treatment": "فیزیوتراپی و کاردرمانی", "priority": "متوسط"},
                {"type": "تجربی", "treatment": "ایمونوتراپی و درمان‌های نوین", "priority": "پایین"}
            ],
            "meningioma": [
                {"type": "جراحی", "treatment": "رزکسیون کامل جراحی", "priority": "بالا"},
                {"type": "رادیوتراپی", "treatment": "رادیوتراپی استریوتاکتیک (در موارد عدم امکان جراحی)", "priority": "متوسط"},
                {"type": "رادیوتراپی", "treatment": "گاما نایف (Gamma Knife)", "priority": "متوسط"},
                {"type": "مشاهده", "treatment": "پایش فعال در موارد آسیمپتوماتیک کوچک", "priority": "متوسط"},
                {"type": "حمایتی", "treatment": "کورتیکواستروئیدها برای کنترل ادم", "priority": "متوسط"},
                {"type": "حمایتی", "treatment": "داروهای ضدتشنج", "priority": "متوسط"},
                {"type": "هورمونی", "treatment": "بررسی و کنترل هورمون‌های جنسی", "priority": "پایین"},
                {"type": "توانبخشی", "treatment": "فیزیوتراپی در صورت نقص عصبی", "priority": "متوسط"}
            ],
            "pituitary": [
                {"type": "جراحی", "treatment": "رزکسیون ترانس‌اسفنوئیدال", "priority": "بالا"},
                {"type": "دارویی", "treatment": "آگونیست‌های دوپامین (برومکریپتین، کابرگولین)", "priority": "بالا"},
                {"type": "دارویی", "treatment": "آنالوگ‌های سوماتواستاتین", "priority": "متوسط"},
                {"type": "دارویی", "treatment": "آنتاگونیست‌های رسپتور هورمون رشد", "priority": "متوسط"},
                {"type": "رادیوتراپی", "treatment": "رادیوتراپی استریوتاکتیک", "priority": "متوسط"},
                {"type": "هورمونی", "treatment": "جایگزینی هورمونی در صورت نقص", "priority": "بالا"},
                {"type": "مشاهده", "treatment": "پایش فعال میکروآدنوم‌های غیرفعال", "priority": "متوسط"},
                {"type": "حمایتی", "treatment": "درمان عوارض هورمونی", "priority": "بالا"}
            ],
            "notumor": [
                {"type": "تشخیصی", "treatment": "بررسی علل دیگر علائم", "priority": "بالا"},
                {"type": "دارویی", "treatment": "مسکن‌ها برای کنترل سردرد", "priority": "متوسط"},
                {"type": "سبک زندگی", "treatment": "مدیریت استرس و تکنیک‌های آرام‌سازی", "priority": "بالا"},
                {"type": "دارویی", "treatment": "داروهای پیشگیری از میگرن", "priority": "متوسط"},
                {"type": "روانشناختی", "treatment": "مشاوره روانشناختی برای اضطراب", "priority": "متوسط"},
                {"type": "پیگیری", "treatment": "MRI پیگیری در فواصل مناسب", "priority": "متوسط"},
                {"type": "سبک زندگی", "treatment": "تنظیم خواب و تغذیه", "priority": "بالا"},
                {"type": "فیزیکی", "treatment": "ورزش منظم و فعالیت بدنی", "priority": "متوسط"}
            ]
        }
        treatments = base_treatments.get(tumor_type, base_treatments["notumor"])
    return treatments

def get_prevention_methods(tumor_type, patient_data):
    """
    روش‌های پیشگیری و کاهش ریسک
    """
    base_methods = {
        "glioma": [
            "محدود کردن مواجهه با پرتوهای یونیزان غیرضروری",
            "اجتناب از استفاده بیش از حد از تلفن همراه (هنوز بحث‌برانگیز)",
            "حفظ سبک زندگی سالم و تغذیه متعادل",
            "کنترل استرس و داشتن خواب کافی",
            "اجتناب از دخانیات و مواد شیمیایی مضر",
            "معاینات منظم در صورت سابقه خانوادگی",
            "گزارش سریع علائم عصبی غیرطبیعی"
        ],
        "meningioma": [
            "محدود کردن مواجهه با پرتوهای یونیزان",
            "مدیریت مناسب هورمون‌های جنسی (مشورت با پزشک)",
            "کنترل منظم فشار خون",
            "حفظ وزن مناسب",
            "تغذیه غنی از آنتی‌اکسیدان‌ها",
            "ورزش منظم",
            "کنترل استرس"
        ],
        "pituitary": [
            "مراجعه به پزشک در صورت علائم هورمونی غیرطبیعی",
            "کنترل منظم هورمون‌ها در صورت سابقه خانوادگی",
            "مدیریت استرس",
            "حفظ وزن مناسب",
            "تغذیه متعادل",
            "خواب کافی و منظم",
            "اجتناب از داروهای غیرضروری که روی هورمون‌ها تأثیر می‌گذارند"
        ],
        "notumor": [
            "مدیریت استرس و تکنیک‌های آرام‌سازی",
            "خواب کافی و منظم (7-8 ساعت)",
            "ورزش منظم",
            "تغذیه متعادل و کاهش کافئین",
            "اجتناب از الکل و دخانیات",
            "مراجعه به پزشک برای سردردهای مداوم",
            "کنترل فشار خون",
            "محدود کردن زمان استفاده از صفحات نمایش"
        ]
    }
    
    return base_methods.get(tumor_type, base_methods["notumor"])

def get_suggested_tests(tumor_type, patient_data):
    """
    آزمایشات و بررسی‌های پیشنهادی
    """
    base_tests = {
        "glioma": [
            "MRI مغز با و بدون کنتراست",
            "CT اسکن مغز",
            "بیوپسی یا رزکسیون جراحی برای تشخیص قطعی",
            "آزمایش پاتولوژی مولکولی تومور",
            "بررسی جهش IDH1/IDH2",
            "تست کودلیشن 1p/19q",
            "مطالعه متیلاسیون MGMT",
            "معاینه عصبی کامل",
            "آزمایشات خونی پایه",
            "ارزیابی عملکرد کبد و کلیه قبل از درمان"
        ],
        "meningioma": [
            "MRI مغز با کنتراست",
            "CT اسکن برای بررسی تهاجم استخوانی",
            "آنژیوگرافی در موارد خاص",
            "بیوپسی در موارد مشکوک",
            "بررسی پاتولوژی درجه‌بندی تومور",
            "معاینه عصبی کامل",
            "بررسی میدان بینایی",
            "آزمایشات هورمونی (در زنان)",
            "ارزیابی قبل از جراحی"
        ],
        "pituitary": [
            "MRI هیپوفیز با کنتراست",
            "آزمایشات هورمونی کامل:",
            "- پرولاکتین",
            "- هورمون رشد (GH) و IGF-1",
            "- ACTH و کورتیزول",
            "- TSH و T4",
            "- LH و FSH",
            "- تستوسترون (مردان) یا استروژن (زنان)",
            "بررسی میدان بینایی",
            "تست تحریک و تثبیط هورمونی",
            "آزمایش ادرار 24 ساعته برای کورتیزول",
            "معاینه چشم‌پزشکی"
        ],
        "notumor": [
            "MRI مغز برای قطع احتمال تومور",
            "CT اسکن در صورت نیاز",
            "آزمایشات خونی پایه:",
            "- CBC",
            "- بیوشیمی",
            "- TSH",
            "- ویتامین B12 و فولات",
            "بررسی فشار خون",
            "معاینه عصبی",
            "ارزیابی روانشناختی در صورت نیاز",
            "بررسی علل دیگر سردرد"
        ]
    }
    
    tests = base_tests.get(tumor_type, base_tests["notumor"])
    
    # اضافه کردن آزمایشات ویژه بر اساس سابقه بیمار
    if 'دیابت' in patient_data.get('medical_history', []):
        tests.extend(['HbA1c', 'گلوکز ناشتا', 'بررسی کنترل دیابت'])
    
    if 'بیماری قلبی' in patient_data.get('medical_history', []):
        tests.extend(['ECG', 'اکوکاردیوگرافی', 'ارزیابی ریسک قلبی'])
    
    return tests

def generate_diagnostic_findings(prediction, patient_data):
    """
    تولید یافته‌های تشخیصی براساس نتیجه مدل
    """
    tumor_diseases = get_brain_tumor_diseases()
    tumor_info = tumor_diseases.get(prediction, {})
    persian_name = tumor_info.get('persian_name', prediction)
    
    if prediction == "glioma":
        findings = f"""
        تحلیل تصویری: تصویر MRI ارائه شده نشان‌دهنده ضایعه‌ای است که ویژگی‌های مطرح‌کننده {persian_name} را دارد.
        
        یافته‌های کلیدی:
        - وجود توده‌ای با حدود نامشخص در پارانشیم مغزی
        - سیگنال ناهمگن و پیچیده در توالی‌های مختلف MRI
        - احتمال وجود ادم اطراف تومور
        - ویژگی‌های تقویت کنندگی متغیر
        
        با توجه به سن بیمار ({patient_data.get('age', 'نامشخص')} سال) و علائم کلینیکی شامل {', '.join(patient_data.get('symptoms', []))}, 
        این یافته‌ها با تشخیص گلیوما سازگار است. مدت زمان {patient_data.get('symptom_duration', 'نامشخص')} علائم 
        با روند پیشرونده این نوع تومور همخوانی دارد.
        """
    
    elif prediction == "meningioma":
        findings = f"""
        تحلیل تصویری: تصویر MRI نشان‌دهنده ضایعه‌ای است که مشخصات {persian_name} را دارد.
        
        یافته‌های کلیدی:
        - وجود توده‌ای با حدود نسبتاً مشخص
        - موقعیت خارج پارانشیمی یا اتصال به دورا ماتر
        - تقویت کنندگی معمولاً یکنواخت
        - احتمال وجود Dural tail
        
        با توجه به جنسیت ({patient_data.get('gender', 'نامشخص')}) و سن بیمار، که مننژیوما در زنان میانسال شایع‌تر است،
        و علائم {', '.join(patient_data.get('symptoms', []))}, تشخیص مننژیوما بسیار محتمل است.
        """
    
    elif prediction == "pituitary":
        findings = f"""
        تحلیل تصویری: تصویر MRI ناحیه سلار نشان‌دهنده تغییراتی در غده هیپوفیز است که مطرح‌کننده {persian_name} می‌باشد.
        
        یافته‌های کلیدی:
        - تغییر در اندازه یا شکل غده هیپوفیز
        - ممکن است میکروآدنوم (کمتر از 1 سانتی‌متر) یا ماکروآدنوم باشد
        - تقویت کنندگی متغیر بسته به نوع آدنوم
        - در موارد بزرگ، احتمال فشار بر کیازم اپتیک
        
        علائم کلینیکی شامل {', '.join(patient_data.get('symptoms', []))} و مدت زمان {patient_data.get('symptom_duration', 'نامشخص')}
        با اختلالات هورمونی ناشی از آدنوم هیپوفیز سازگار است.
        """
    
    else:  # notumor
        findings = f"""
        تحلیل تصویری: تصویر MRI ارائه شده ساختار طبیعی مغز را نشان می‌دهد و هیچ علامتی از وجود تومور مشاهده نمی‌شود.
        
        یافته‌های کلیدی:
        - ساختار آناتومیکی طبیعی مغز
        - عدم وجود توده یا ضایعه قابل تشخیص
        - بطن‌های طبیعی
        - عدم ادم یا اثر فضاگیر
        
        با وجود عدم وجود تومور، علائم بیمار شامل {', '.join(patient_data.get('symptoms', []))} 
        نیازمند بررسی علل دیگر است. ممکن است علائم ناشی از سردردهای اولیه، مشکلات عروقی، 
        یا عوامل دیگر باشد.
        """
    
    return findings.strip()

def generate_treatment_approach(prediction, patient_data):
   """
   تولید رویکرد درمانی جامع
   """
   tumor_diseases = get_brain_tumor_diseases()
   tumor_info = tumor_diseases.get(prediction, {})
   persian_name = tumor_info.get('persian_name', prediction)
   risk_level = tumor_info.get('risk_level', 'متوسط')
   
   if prediction == "glioma":
       approach = f"""
       رویکرد درمانی برای {persian_name}:
       
       با توجه به ماهیت بدخیم این تومور و سطح ریسک {risk_level}، رویکرد درمانی مولتی‌مودال شامل:
       
       1. مرحله اول - جراحی فوری:
          - رزکسیون جراحی حداکثری با حفظ عملکرد عصبی
          - بیوپسی فوری برای تشخیص قطعی
          - کنترل فشار داخل جمجمه
       
       2. مرحله دوم - رادیوتراپی:
          - شروع رادیوتراپی ظرف 2-6 هفته پس از جراحی
          - دوز کل 60 گری در 30 فراکشن
          - رادیوتراپی کنفورمال یا IMRT
       
       3. مرحله سوم - شیمی‌درمانی:
          - تمازولومید همزمان با رادیوتراپی
          - تمازولومید نگهدارنده به مدت 6-12 ماه
          - تنظیم دوز بر اساس تحمل بیمار
       
       4. درمان‌های حمایتی:
          - کورتیکواستروئیدها برای کنترل ادم مغزی
          - داروهای ضدتشنج در صورت نیاز
          - پروفیلاکسی PCP با تریمتوپریم-سولفامتوکسازول
       
       5. پیگیری و نظارت:
          - MRI هر 2-3 ماه در سال اول
          - بررسی عوارض درمان
          - ارزیابی کیفیت زندگی
       """
   
   elif prediction == "meningioma":
       approach = f"""
       رویکرد درمانی برای {persian_name}:
       
       با توجه به ماهیت معمولاً خوش‌خیم این تومور و سطح ریسک {risk_level}:
       
       1. ارزیابی اولیه:
          - تعیین درجه تومور (Grade I, II, III)
          - بررسی موقعیت و قابلیت رزکسیون
          - ارزیابی عملکرد عصبی
       
       2. درمان بر اساس شرایط:
          
          الف) مننژیوم‌های کوچک و بدون علامت:
          - پایش فعال با MRI هر 6-12 ماه
          - عدم نیاز به درمان فوری
          
          ب) مننژیوم‌های علامت‌دار یا بزرگ:
          - جراحی رزکسیون کامل (Simpson Grade I-II)
          - حفظ ساختارهای عصبی مهم
          
          ج) مننژیوم‌های غیرقابل رزکسیون:
          - رادیوجراحی استریوتاکتیک
          - گاما نایف یا CyberKnife
          - رادیوتراپی فراکشن‌بندی شده
       
       3. درمان‌های حمایتی:
          - کنترل ادم با استروئیدها
          - داروهای ضدتشنج در صورت وجود تشنج
          - مدیریت درد
       
       4. پیگیری:
          - MRI در فواصل 6-12 ماهه
          - پایش علائم عصبی
          - ارزیابی عود در طولانی‌مدت
       """
   
   elif prediction == "pituitary":
       approach = f"""
       رویکرد درمانی برای {persian_name}:
       
       بر اساس نوع آدنوم (عملکردی/غیرعملکردی) و اندازه:
       
       1. میکروآدنوم‌های غیرعملکردی:
          - پایش فعال با MRI سالانه
          - بررسی عملکرد هورمونی
          - مداخله فقط در صورت رشد یا علامت
       
       2. ماکروآدنوم‌ها:
          - جراحی ترانس‌اسفنوئیدال
          - حفظ بافت هیپوفیز سالم
          - رفع فشار بر کیازم اپتیک
       
       3. آدنوم‌های عملکردی:
          
          الف) پرولاکتینوما:
          - درمان دارویی اولیه با آگونیست‌های دوپامین
          - کابرگولین یا برومکریپتین
          - جراحی در موارد مقاوم به دارو
          
          ب) آدنوم‌های ترشح‌کننده هورمون رشد:
          - جراحی اولیه
          - آنالوگ‌های سوماتواستاتین
          - آنتاگونیست‌های رسپتور GH
          
          ج) بیماری کوشینگ:
          - جراحی ترانس‌اسفنوئیدال
          - کنترل کورتیزول قبل از جراحی
          - درمان‌های دارویی تکمیلی
       
       4. مدیریت عوارض:
          - جایگزینی هورمونی در صورت نقص
          - درمان دیابت بی‌مزه
          - کنترل اختلالات الکترولیتی
       
       5. پیگیری:
          - MRI و آزمایشات هورمونی منظم
          - معاینه میدان بینایی
          - پایش عوارض جانبی درمان
       """
   
   else:  # notumor
       approach = f"""
       رویکرد مدیریتی برای علائم با تصویربرداری طبیعی:
       
       1. بررسی علل جایگزین:
          - سردردهای اولیه (میگرن، تنشی، خوشه‌ای)
          - اختلالات عروقی مغزی
          - هیپرتانسیون داخل جمجمه کاذب
          - التهابات یا عفونت‌ها
          - اختلالات متابولیک
       
       2. مدیریت علامتی:
          - مسکن‌های مناسب برای سردرد
          - تریپتان‌ها برای میگرن
          - پیشگیری از میگرن با بتابلوکرها یا آنتی‌کانولسانت‌ها
          - مدیریت فشار خون
       
       3. تغییرات سبک زندگی:
          - تنظیم الگوی خواب
          - کاهش استرس و تکنیک‌های آرام‌سازی
          - ورزش منظم و تناسب اندام
          - اجتناب از عوامل تریگرکننده سردرد
          - تغذیه منظم و متعادل
       
       4. درمان‌های تکمیلی:
          - فیزیوتراپی برای سردردهای تنشی
          - ماساژ درمانی
          - طب سوزنی
          - تکنیک‌های مدیتیشن و یوگا
       
       5. پیگیری و نظارت:
          - بازبینی علائم در فواصل منظم
          - MRI مجدد در صورت تغییر یا تشدید علائم
          - ارجاع به متخصصین مربوطه در صورت نیاز
          - آموزش بیمار و خانواده
       
       6. نشانه‌های خطر برای مراجعه فوری:
          - سردرد ناگهانی و شدید
          - تب همراه با سردرد
          - اختلالات بینایی
          - ضعف یا بی‌حسی
          - تشنج
          - تغییرات شخصیتی شدید
       """
   
   # اضافه کردن توصیه‌های ویژه بر اساس سن و شرایط بیمار
   age = patient_data.get('age', 0)
   if age > 70:
       approach += f"""
       
       توجهات ویژه سالمندان:
       - تنظیم دوز داروها با در نظر گیری عملکرد کلیه
       - ارزیابی ریسک-فایده دقیق‌تر برای درمان‌های تهاجمی
       - توجه ویژه به کیفیت زندگی
       - پیشگیری از عوارض و عفونت‌ها
       - نظارت دقیق‌تر بر عوارض جانبی
       """
   
   if 'دیابت' in patient_data.get('medical_history', []):
       approach += """
       
       توجهات ویژه دیابتی‌ها:
       - مراقبت در استفاده از کورتیکواستروئیدها
       - کنترل دقیق قند خون
       - هماهنگی با اندوکرینولوژیست
       - پیشگیری از عفونت‌ها
       """
   
   if 'بیماری قلبی' in patient_data.get('medical_history', []):
       approach += """
       
       توجهات ویژه بیماران قلبی:
       - ارزیابی ریسک قلبی-عروقی قبل از جراحی
       - کنسولت قلب قبل از درمان
       - مراقبت در استفاده از داروهای خاص
       - نظارت بر ریتم قلب
       """
   
   return approach.strip()

# توابع کمکی اضافی برای تکمیل سیستم

def get_prognosis_details(prediction, patient_data):
   """
   جزئیات پیش‌آگهی بر اساس نوع تومور و شرایط بیمار
   """
   tumor_diseases = get_brain_tumor_diseases()
   tumor_info = tumor_diseases.get(prediction, {})
   
   base_prognosis = {
       "glioma": {
           "short_term": "متغیر بسته به درجه و موقعیت تومور",
           "long_term": "پیش‌آگهی کلی محتاط، بازماندگی 5 ساله متغیر از 5% تا 90%",
           "factors": [
               "درجه تومور (مهمترین فاکتور)",
               "سن بیمار",
               "وضعیت عملکردی",
               "میزان رزکسیون جراحی",
               "پاسخ به درمان",
               "جهش‌های مولکولی (IDH, MGMT)"
           ]
       },
       "meningioma": {
           "short_term": "معمولاً بسیار خوب",
           "long_term": "پیش‌آگهی عالی در اکثر موارد، نرخ بازگشت کم",
           "factors": [
               "درجه تومور",
               "کامل بودن رزکسیون",
               "موقعیت تومور",
               "سن بیمار",
               "وضعیت عملکردی"
           ]
       },
       "pituitary": {
           "short_term": "بسیار خوب",
           "long_term": "پیش‌آگهی عالی با درمان مناسب",
           "factors": [
               "نوع آدنوم",
               "اندازه تومور",
               "میزان تهاجم",
               "پاسخ به درمان دارویی",
               "وضعیت هورمونی"
           ]
       },
       "notumor": {
           "short_term": "بسیار خوب",
           "long_term": "عالی با مدیریت مناسب علل زمینه‌ای",
           "factors": [
               "علت اصلی علائم",
               "پاسخ به درمان",
               "عوامل سبک زندگی",
               "عوامل استرس‌زا"
           ]
       }
   }
   
   return base_prognosis.get(prediction, base_prognosis["notumor"])

def get_followup_schedule(prediction, patient_data):
   """
   برنامه پیگیری بر اساس نوع تومور
   """
   schedules = {
       "glioma": {
           "imaging": "MRI هر 2-3 ماه در سال اول، سپس هر 6 ماه",
           "clinical": "معاینه عصبی ماهانه در 6 ماه اول",
           "labs": "CBC و LFT قبل از هر سیکل شیمی‌درمانی",
           "other": "ارزیابی کیفیت زندگی، معاینه چشم، بررسی عوارض"
       },
       "meningioma": {
           "imaging": "MRI در 3 ماه اول، سپس سالانه",
           "clinical": "معاینه عصبی هر 6 ماه",
           "labs": "بررسی هورمونی در موارد خاص",
           "other": "ارزیابی علائم، کنترل تشنج"
       },
       "pituitary": {
           "imaging": "MRI در 3-6 ماه اول، سپس سالانه",
           "clinical": "معاینه اندوکرین هر 3-6 ماه",
           "labs": "آزمایشات هورمونی منظم",
           "other": "معاینه میدان بینایی، بررسی عوارض جانبی"
       },
       "notumor": {
           "imaging": "MRI فقط در صورت تغییر علائم",
           "clinical": "معاینه هر 6-12 ماه",
           "labs": "بر اساس علل زمینه‌ای",
           "other": "پیگیری درمان علامتی، بررسی عوامل محیطی"
       }
   }
   
   return schedules.get(prediction, schedules["notumor"])

# --- روت‌ها ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = request.form.get('age')
        gender = request.form.get('gender')
        symptoms_list = request.form.getlist('symptoms')
        symptoms_text = request.form.get('symptoms', '')
        symptoms = ', '.join(symptoms_list)
        if symptoms_text:
            symptoms = symptoms + ('، ' if symptoms else '') + symptoms_text
        file = request.files.get('image')
        if not (age and gender and symptoms and file and allowed_file(file.filename)):
            flash('لطفاً همه فیلدها را کامل کنید و تصویر مناسب انتخاب کنید.')
            return redirect(request.url)
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        prediction = predict_image(filename)
        info = f"Age: {age}, Gender: {gender}, Tumor: {prediction}"
        advice_groq = get_medical_advice(symptoms, info)
        # اطلاعات دانش‌بنیان
        diseases = get_brain_tumor_diseases()
        disease_info = diseases.get(prediction, {})
        features = get_diagnostic_features(prediction)
        treatments = get_treatments(prediction, {"age": int(age) if age else 0, "symptoms": symptoms_list})
        prevention = get_prevention_methods(prediction, {"symptoms": symptoms_list})
        tests = get_suggested_tests(prediction, {"age": int(age) if age else 0, "symptoms": symptoms_list})
        image_url = url_for('uploaded_file', filename=file.filename)
        return render_template('result.html', prediction=prediction, advice=advice_groq, age=age, gender=gender, symptoms=symptoms, disease_info=disease_info, features=features, treatments=treatments, prevention=prevention, tests=tests, image_url=image_url)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return flask.send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- اجرای برنامه ---
if __name__ == '__main__':
    app.run(debug=True) 