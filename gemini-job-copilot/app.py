"""
Job Application Copilot
========================
ต่อยอดจาก Lab GSP1150 (Introduction to Gemini 3 / Google Gen AI SDK)

แอปนี้รับ "ประกาศงาน" + "เรซูเม่" + "ชื่อบริษัท" แล้ว:
  1. ค้นข้อมูลบริษัทล่าสุดด้วย Grounding (Google Search)
  2. คำนวณอายุงานจริงจากเรซูเม่ด้วย Function Calling
  3. วิเคราะห์ความเหมาะสมออกมาเป็น Structured Output (JSON ตาม schema)
  4. เปิดโหมดซ้อมสัมภาษณ์แบบ Multi-turn Chat + Streaming (โจทย์เสริม)
  5. เปรียบเทียบผลลัพธ์ระหว่าง thinking_level ต่างกัน (โจทย์เสริม, --compare-thinking)

รายละเอียด/เหตุผลการออกแบบ prompt + temperature อยู่ใน SPEC.md และ README.md
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
API_KEY = os.getenv("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# System Instructions (บทบาทของโมเดล)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION_ANALYST = (
    "คุณคือนักวิเคราะห์ HR มืออาชีพที่ตรงไปตรงมาแต่ให้กำลังใจ "
    "วิเคราะห์อย่างเป็นกลางโดยอิงจากหลักฐานในเรซูเม่และประกาศงานที่ได้รับเท่านั้น "
    "ห้ามเดาหรือสมมติข้อมูลที่ไม่มีอยู่ในเอกสารที่ให้มา "
    "ตอบกลับเป็น JSON ตาม schema ที่กำหนดเท่านั้น ห้ามมีข้อความอื่นนอก JSON"
)

SYSTEM_INSTRUCTION_COACH = (
    "คุณคือโค้ชเตรียมสัมภาษณ์งานที่เป็นกันเองและมีประสบการณ์สูง "
    "ถามคำถามสัมภาษณ์ทีละข้อ รอฟังคำตอบผู้ใช้ก่อน แล้วให้ feedback สั้น กระชับ ตรงประเด็นทันที "
    "ก่อนจะถามคำถามถัดไป ให้กำลังใจผู้ใช้แต่ก็ชี้จุดที่ควรปรับปรุงตรง ๆ "
    "ใช้ภาษาไทยเป็นหลัก น้ำเสียงเป็นกันเองเหมือนโค้ชตัวจริง"
)

SYSTEM_INSTRUCTION_RESEARCHER = (
    "คุณคือผู้ช่วยรีเสิร์ชข้อมูลบริษัทสำหรับผู้เตรียมตัวสัมภาษณ์งาน "
    "ตอบสั้น กระชับ เป็น bullet point อ้างอิงข้อมูลที่ใหม่และตรวจสอบได้เท่านั้น"
)

SYSTEM_INSTRUCTION_EXTRACTOR = (
    "คุณคือผู้ช่วยดึงข้อมูลประวัติการทำงานจากเรซูเม่อย่างละเอียดและแม่นยำ "
    "เมื่อพบช่วงเวลาการทำงาน ให้เรียกฟังก์ชัน calc_years_experience เสมอ "
    "ห้ามคำนวณอายุงานเอง ให้ฟังก์ชันเป็นผู้คำนวณ"
)


# ---------------------------------------------------------------------------
# Structured Output Schema (สำหรับผลวิเคราะห์)
# ---------------------------------------------------------------------------

ANALYSIS_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "match_score": {
            "type": "INTEGER",
            "description": "คะแนนความเหมาะสมของผู้สมัครกับตำแหน่งงาน 0-100",
        },
        "summary": {
            "type": "STRING",
            "description": "สรุปภาพรวมความเหมาะสม 2-3 ประโยค",
        },
        "strengths": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "จุดแข็งของผู้สมัครที่ตรงกับประกาศงาน",
        },
        "gaps": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "จุดที่ผู้สมัครยังขาดหรือควรพัฒนาเพิ่ม",
        },
        "recommended_skills_to_learn": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "ทักษะที่แนะนำให้พัฒนาเพิ่มก่อนสมัคร/สัมภาษณ์",
        },
        "estimated_years_experience": {
            "type": "NUMBER",
            "description": "อายุงานรวมโดยประมาณ (ปี) ตามที่คำนวณได้จาก function calling",
        },
        "interview_questions": {
            "type": "ARRAY",
            "description": "ชุดคำถามสัมภาษณ์ที่คาดว่าจะถูกถามสำหรับตำแหน่งนี้",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "category": {"type": "STRING", "description": "หมวดคำถาม เช่น Technical, Behavioral"},
                    "question": {"type": "STRING"},
                    "tip": {"type": "STRING", "description": "เทคนิคการตอบคำถามนี้"},
                },
                "required": ["category", "question", "tip"],
            },
        },
    },
    "required": [
        "match_score",
        "summary",
        "strengths",
        "gaps",
        "recommended_skills_to_learn",
        "interview_questions",
    ],
}


# ---------------------------------------------------------------------------
# Function Calling: ฟังก์ชันจริงที่คำนวณอายุงาน (deterministic, ไม่ใช่โมเดลเดา)
# ---------------------------------------------------------------------------

CALC_YEARS_FUNCTION_DECLARATION = types.FunctionDeclaration(
    name="calc_years_experience",
    description=(
        "คำนวณอายุงานรวม (ปี) จากรายการช่วงเวลาทำงานที่ดึงมาจากเรซูเม่ "
        "โมเดลควรเรียกฟังก์ชันนี้ทุกครั้งที่ต้องการทราบอายุงานรวม แทนการคำนวณเอง"
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "periods": {
                "type": "ARRAY",
                "description": "รายการช่วงเวลาทำงานแต่ละงาน",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "start": {
                            "type": "STRING",
                            "description": "วันที่เริ่มงาน รูปแบบ YYYY-MM",
                        },
                        "end": {
                            "type": "STRING",
                            "description": "วันที่สิ้นสุดงาน รูปแบบ YYYY-MM หรือคำว่า 'present' ถ้ายังทำอยู่",
                        },
                    },
                    "required": ["start"],
                },
            }
        },
        "required": ["periods"],
    },
)


def calc_years_experience(periods: list) -> float:
    """ฟังก์ชัน Python จริงที่โมเดลจะเรียกผ่าน Function Calling (deterministic)."""
    total_months = 0
    now = datetime.now()
    for p in periods:
        start_raw = p.get("start")
        end_raw = p.get("end", "present")
        try:
            start = datetime.strptime(start_raw, "%Y-%m")
        except (TypeError, ValueError):
            continue
        if not end_raw or str(end_raw).strip().lower() in ("present", "ปัจจุบัน", "now", ""):
            end = now
        else:
            try:
                end = datetime.strptime(end_raw, "%Y-%m")
            except ValueError:
                end = now
        months = (end.year - start.year) * 12 + (end.month - start.month)
        total_months += max(months, 0)
    return round(total_months / 12, 1)


# ---------------------------------------------------------------------------
# Client / helpers
# ---------------------------------------------------------------------------

def get_client() -> genai.Client:
    if not API_KEY:
        print("❌ ไม่พบ GEMINI_API_KEY กรุณาตั้งค่าในไฟล์ .env (ดู .env.example)")
        sys.exit(1)
    return genai.Client(api_key=API_KEY)


def read_input(value: str) -> str:
    """ถ้า value เป็น path ของไฟล์ที่มีอยู่จริง ให้อ่านไฟล์ ไม่งั้นถือว่าเป็น text ตรง ๆ"""
    if value and os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as f:
            return f.read()
    return value


# ---------------------------------------------------------------------------
# ฟีเจอร์ 1: Grounding ด้วย Google Search
# ---------------------------------------------------------------------------

def research_company(client: genai.Client, company_name: str) -> str:
    search_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION_RESEARCHER,
        tools=[search_tool],
        temperature=0.3,  # ต้องการข้อเท็จจริง แต่ยอมให้สรุปลื่นไหลได้บ้าง
    )
    prompt = (
        f"ค้นข้อมูลล่าสุดเกี่ยวกับบริษัท '{company_name}' ที่เป็นประโยชน์ต่อผู้เตรียมตัวสัมภาษณ์งาน "
        "เช่น ธุรกิจหลัก ข่าวล่าสุด ทิศทางธุรกิจ วัฒนธรรมองค์กร สรุปเป็น bullet point 5-8 ข้อ ภาษาไทย"
    )
    resp = client.models.generate_content(model=MODEL, contents=prompt, config=config)
    return resp.text or ""


# ---------------------------------------------------------------------------
# ฟีเจอร์ 2: Function Calling
# ---------------------------------------------------------------------------

def extract_experience_years(client: genai.Client, resume_text: str):
    tool = types.Tool(function_declarations=[CALC_YEARS_FUNCTION_DECLARATION])
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION_EXTRACTOR,
        tools=[tool],
        temperature=0.0,  # ต้องการความแม่นยำ/deterministic เต็มที่
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=(
                "นี่คือเรซูเม่:\n"
                f"{resume_text}\n\n"
                "กรุณาดึงช่วงเวลาการทำงานทั้งหมด (start, end ในรูปแบบ YYYY-MM) "
                "แล้วเรียกฟังก์ชัน calc_years_experience เพื่อคำนวณอายุงานรวม"
            ))],
        )
    ]

    resp = client.models.generate_content(model=MODEL, contents=contents, config=config)

    years = None
    for candidate in resp.candidates or []:
        for part in candidate.content.parts:
            if getattr(part, "function_call", None) and part.function_call.name == "calc_years_experience":
                args = dict(part.function_call.args)
                years = calc_years_experience(args.get("periods", []))

                # ส่งผลลัพธ์ฟังก์ชันกลับให้โมเดลเพื่อความสมบูรณ์ของ flow (ไม่บังคับต้องใช้ต่อ)
                contents.append(candidate.content)
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name="calc_years_experience",
                                response={"total_years": years},
                            )
                        ],
                    )
                )
                client.models.generate_content(model=MODEL, contents=contents, config=config)
                break
        if years is not None:
            break

    return years


# ---------------------------------------------------------------------------
# ฟีเจอร์ 3: Structured Output (JSON ตาม schema)
# ---------------------------------------------------------------------------

def analyze_application(
    client: genai.Client,
    job_text: str,
    resume_text: str,
    company_research: str,
    years_experience,
    thinking_level: str = None,
):
    config_kwargs = dict(
        system_instruction=SYSTEM_INSTRUCTION_ANALYST,
        temperature=0.2,  # ต้องการผลตรวจสอบได้ ทำซ้ำได้ ไม่อยากให้คะแนน/สรุปแกว่งมาก
        response_mime_type="application/json",
        response_schema=ANALYSIS_RESPONSE_SCHEMA,
    )
    if thinking_level:
        # หมายเหตุ: ชื่อพารามิเตอร์/ค่า enum ของ thinking config อาจต่างกันตามเวอร์ชัน SDK
        # ให้ยึดตาม notebook ของ Lab GSP1150 เป็นหลักหากชื่อ parameter เปลี่ยนไป
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

    config = types.GenerateContentConfig(**config_kwargs)

    prompt = f"""
ประกาศงาน:
{job_text}

เรซูเม่ผู้สมัคร:
{resume_text}

ข้อมูลบริษัท (จากการค้นล่าสุด):
{company_research}

อายุงานรวมที่คำนวณได้จริงจากฟังก์ชัน: {years_experience if years_experience is not None else 'ไม่สามารถคำนวณได้'} ปี

กรุณาวิเคราะห์ความเหมาะสมของผู้สมัครกับตำแหน่งงานนี้ และเตรียมชุดคำถามสัมภาษณ์ที่เกี่ยวข้อง
"""
    resp = client.models.generate_content(model=MODEL, contents=prompt, config=config)
    return json.loads(resp.text)


# ---------------------------------------------------------------------------
# โจทย์เสริม A: เปรียบเทียบ thinking_level ต่างกัน
# ---------------------------------------------------------------------------

def compare_thinking_levels(client, job_text, resume_text, company_research, years_experience):
    levels = ["low", "high"]
    print("\n🧪 เปรียบเทียบผลลัพธ์ระหว่าง thinking_level ต่างกัน (โจทย์เสริม)")
    for level in levels:
        print(f"\n--- thinking_level = {level} ---")
        start = time.time()
        try:
            result = analyze_application(
                client, job_text, resume_text, company_research, years_experience,
                thinking_level=level,
            )
            elapsed = time.time() - start
            print(f"⏱️ ใช้เวลา {elapsed:.2f} วินาที")
            print(f"match_score: {result.get('match_score')}")
            print(f"summary: {result.get('summary')}")
        except Exception as e:
            print(f"⚠️ ไม่สามารถรันด้วย thinking_level='{level}' บน SDK/โมเดลนี้ได้: {e}")


# ---------------------------------------------------------------------------
# โจทย์เสริม B: Multi-turn Chat + Streaming (โหมดซ้อมสัมภาษณ์)
# ---------------------------------------------------------------------------

def interview_chat(client: genai.Client, analysis: dict, job_text: str):
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION_COACH,
        temperature=0.7,  # ต้องการความเป็นธรรมชาติ หลากหลาย ไม่ซ้ำซาก
    )
    chat = client.chats.create(model=MODEL, config=config)

    questions = analysis.get("interview_questions", [])
    first_q = questions[0] if questions else {"question": "แนะนำตัวเองหน่อยครับ/ค่ะ"}

    intro = (
        f"บริบทตำแหน่งงาน:\n{job_text}\n\n"
        f"จุดที่ผู้สมัครควรพัฒนา: {analysis.get('gaps')}\n\n"
        f"เริ่มซ้อมสัมภาษณ์โดยถามคำถามแรกนี้ก่อน: {first_q.get('question')}"
    )

    print("\n🎙️  เริ่มโหมดซ้อมสัมภาษณ์ (พิมพ์ 'exit' เพื่อจบการสนทนา)\n")
    print("Coach: ", end="", flush=True)
    for chunk in chat.send_message_stream(intro):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")

    while True:
        try:
            user_input = input("คุณ: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in ("exit", "quit", "q", "ออก"):
            print("จบการซ้อมสัมภาษณ์ 🙌")
            break
        print("Coach: ", end="", flush=True)
        for chunk in chat.send_message_stream(user_input):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Job Application Copilot (Gemini API)")
    parser.add_argument("--job", default="samples/sample_job.txt", help="path ไฟล์ประกาศงาน หรือ text ตรง ๆ")
    parser.add_argument("--resume", default="samples/sample_resume.txt", help="path ไฟล์เรซูเม่ หรือ text ตรง ๆ")
    parser.add_argument("--company", default="Krungthai Bank", help="ชื่อบริษัทที่จะค้นข้อมูลด้วย Grounding")
    parser.add_argument("--interactive", action="store_true", help="เปิดโหมดซ้อมสัมภาษณ์แบบสนทนา")
    parser.add_argument("--compare-thinking", action="store_true", help="เปรียบเทียบผลลัพธ์ระหว่าง thinking_level")
    parser.add_argument("--out", default="output/analysis.json", help="path สำหรับบันทึกผลวิเคราะห์ JSON")
    args = parser.parse_args()

    client = get_client()

    job_text = read_input(args.job)
    resume_text = read_input(args.resume)

    print("🔎 กำลังค้นข้อมูลบริษัท (Grounding: Google Search)...")
    company_research = research_company(client, args.company)
    print(company_research)

    print("\n🧮 กำลังดึงประวัติทำงานและคำนวณอายุงาน (Function Calling)...")
    years = extract_experience_years(client, resume_text)
    print(f"อายุงานรวมที่คำนวณได้: {years} ปี")

    print("\n🧠 กำลังวิเคราะห์ใบสมัคร (Structured Output / JSON Schema)...")
    analysis = analyze_application(client, job_text, resume_text, company_research, years)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"\n✅ บันทึกผลวิเคราะห์ที่ {args.out}\n")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))

    if args.compare_thinking:
        compare_thinking_levels(client, job_text, resume_text, company_research, years)

    if args.interactive:
        interview_chat(client, analysis, job_text)


if __name__ == "__main__":
    main()
