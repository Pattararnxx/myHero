# Job Application Copilot

ต่อยอดจาก **Lab GSP1150 · Introduction to Gemini 3** (Gemini API ผ่าน Google Gen AI SDK / `genai`)

แอป CLI ที่รับ **ประกาศงาน + เรซูเม่ + ชื่อบริษัท** แล้ววิเคราะห์ความเหมาะสมของผู้สมัคร พร้อมชุดคำถามสัมภาษณ์ และเปิดโหมดซ้อมสัมภาษณ์แบบสนทนาจริงได้

ดูสเปกที่เขียนไว้ก่อนลงมือทำที่ [`SPEC.md`](./SPEC.md)

## ฟีเจอร์จาก Lab ที่นำมาใช้

| ฟีเจอร์ | ใช้ตรงไหนในแอป |
|---|---|
| **System Instructions** | 3 บทบาทแยกกัน: นักวิเคราะห์ HR / นักรีเสิร์ชบริษัท / โค้ชสัมภาษณ์ |
| **Structured Output (JSON Schema)** | ผลวิเคราะห์ใบสมัคร (`analyze_application`) ออกมาเป็น JSON ตาม schema คงที่ นำไปต่อยอด (เช่นเก็บลง DB, ทำ dashboard) ได้ทันที |
| **Grounding (Google Search)** | ค้นข้อมูลบริษัทล่าสุดจริง ๆ ก่อนใช้ประกอบการวิเคราะห์ (`research_company`) |
| **Function Calling** | โมเดลดึงช่วงเวลาทำงานจากเรซูเม่ แล้วเรียกฟังก์ชัน Python จริง `calc_years_experience` เพื่อคำนวณอายุงาน (ไม่ปล่อยให้โมเดลเดาเลขเอง) |
| **Multi-turn Chat + Streaming** (โจทย์เสริม) | โหมด `--interactive` ใช้ `chat.send_message_stream()` ซ้อมสัมภาษณ์แบบจำบริบทและตอบ real-time |
| **thinking_level comparison** (โจทย์เสริม) | flag `--compare-thinking` รันวิเคราะห์เดียวกันด้วย `thinking_level="low"` vs `"high"` แล้วเทียบเวลา/ผลลัพธ์ |

## วิธีติดตั้งและรัน

```bash
git clone <repo-url>
cd gemini-job-copilot

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# แก้ .env ใส่ GEMINI_API_KEY ของคุณ (ขอได้ฟรีที่ https://aistudio.google.com/apikey)
```

### รันแบบพื้นฐาน (ใช้ไฟล์ตัวอย่างใน samples/)

```bash
python app.py
```

### รันด้วยไฟล์ของตัวเอง

```bash
python app.py --job path/to/job_posting.txt --resume path/to/my_resume.txt --company "ชื่อบริษัท"
```

### เปิดโหมดซ้อมสัมภาษณ์ (multi-turn chat + streaming)

```bash
python app.py --interactive
```

### เปรียบเทียบ thinking_level

```bash
python app.py --compare-thinking
```

## ตัวอย่าง Input / Output

**Input:** `samples/sample_job.txt` (ประกาศงาน Data Analyst) + `samples/sample_resume.txt` (เรซูเม่ 2 งาน รวม ~2 ปี)

**Output ตัวอย่าง** (`output/analysis.json`, รูปแบบจริง — ค่าตัวเลขจะต่างกันไปตามการรันจริง):

```json
{
  "match_score": 78,
  "summary": "ผู้สมัครมีพื้นฐาน SQL และ Power BI ตรงกับที่ประกาศงานต้องการ แต่ยังขาดประสบการณ์ Machine Learning เชิงลึก",
  "strengths": [
    "มีประสบการณ์เขียน SQL และสร้าง Dashboard ด้วย Power BI ตรงกับตำแหน่ง",
    "มีประสบการณ์นำเสนอข้อมูลต่อผู้บริหารตามที่ประกาศงานต้องการ"
  ],
  "gaps": [
    "ประสบการณ์ Machine Learning ยังอยู่ระดับเริ่มต้น ขณะที่ประกาศงานระบุว่าเป็นข้อได้เปรียบ"
  ],
  "recommended_skills_to_learn": [
    "สถิติเชิงลึกสำหรับ Machine Learning",
    "การทำ A/B Testing"
  ],
  "estimated_years_experience": 2.1,
  "interview_questions": [
    {
      "category": "Technical",
      "question": "ช่วยเล่าขั้นตอนการสร้าง Dashboard ใน Power BI ที่คุณเคยทำให้ฟังหน่อย",
      "tip": "เน้นเล่าปัญหาทางธุรกิจที่ dashboard นั้นช่วยแก้ ไม่ใช่แค่ฟีเจอร์ที่ใช้"
    }
  ]
}
```

## การตั้งค่า Temperature และเหตุผล

| ส่วนของแอป | Temperature | เหตุผล |
|---|---|---|
| Grounding (ค้นข้อมูลบริษัท) | `0.3` | ต้องการคำตอบที่อิงข้อเท็จจริงเป็นหลัก แต่ยอมให้สรุปเป็นประโยคที่ลื่นไหลอ่านง่ายได้บ้าง |
| Function calling (ดึงช่วงเวลาทำงาน) | `0.0` | งานนี้ต้องการความแม่นยำ/deterministic เต็มที่ ไม่ต้องการ "ความคิดสร้างสรรค์" ใด ๆ จากโมเดล เพราะการคำนวณจริงทำโดยฟังก์ชัน Python ไม่ใช่โมเดล |
| Structured output (วิเคราะห์ใบสมัคร) | `0.2` | ต้องการผลที่ตรวจสอบได้และไม่แกว่งมากระหว่างการรันซ้ำ (คะแนน/สรุปควรสอดคล้องกันถ้า input เดิม) แต่ยังเปิดพื้นที่เล็กน้อยให้ปรับสำนวนได้ |
| Interview chat coach | `0.7` | ต้องการบุคลิกที่เป็นธรรมชาติ คำถาม/โทนเสียงหลากหลายไม่ซ้ำซากเหมือนคุยกับคนจริง |

## ส่วนที่ใช้ AI ช่วยเขียน (ตามข้อกำหนดของวิชา)

- ใช้ Claude ช่วยร่างโครงสร้างโค้ด `app.py`, schema ของ structured output, ฟังก์ชัน `calc_years_experience`, และร่างเอกสาร `SPEC.md` / `README.md` นี้ตาม prompt design ที่ผู้เขียนกำหนดโจทย์และ requirement เอง
- ผู้เขียนเป็นผู้กำหนด use case, schema, system instructions, และตรวจสอบ/ปรับโค้ดก่อนใช้งานจริง

## โครงสร้างไฟล์

```
gemini-job-copilot/
├── README.md
├── SPEC.md
├── app.py
├── requirements.txt
├── .env.example
├── samples/
│   ├── sample_job.txt
│   └── sample_resume.txt
└── output/          # ผลลัพธ์ JSON จะถูกสร้างที่นี่หลังรัน
```

## หมายเหตุ

- `thinking_config` / `thinking_level` เป็นฟีเจอร์ของ Gemini 3 ที่ชื่อ parameter หรือ enum อาจเปลี่ยนแปลงตามเวอร์ชัน SDK — ถ้ารันแล้ว error ให้ตรวจสอบ syntax ล่าสุดจาก notebook ของ Lab GSP1150 หรือ [เอกสาร Gemini API](https://ai.google.dev/gemini-api/docs)
- โมเดล default ตั้งเป็น `gemini-3-pro-preview` ผ่าน `.env` (`GEMINI_MODEL`) — เปลี่ยนเป็นรุ่นอื่นได้ตามสิทธิ์การเข้าถึงของ API key
