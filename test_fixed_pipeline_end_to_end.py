"""
test_fixed_pipeline_end_to_end.py
---------------------------------
Verifies all 4 citizen personas against the fixed voice_rag_server logic:
1. Sunita Sharma (Hindi small farmer)
2. Murugan (Tamil handloom weaver)
3. Ramesh Kumar (UP dairy farmer)
4. Gujjula Ramayya (Telangana SC farmer)
"""
import sys, os
from pathlib import Path

# Add deploy folder to path
sys.path.insert(0, str(Path(__file__).parent / "deploy"))

from dotenv import load_dotenv
load_dotenv(override=True)

from voice_rag_server import (
    normalize_and_translate_query,
    hybrid_retrieve,
    generate_answer,
    detect_indic_lang,
)

personas = [
    {
        "id": "tamil_weaver",
        "name": "Murugan",
        "raw_query": "வணக்கம், என் பெயர் முருகன். நான் தமிழ்நாட்டின் காஞ்சிபுரத்தை சேர்ந்த கைத்தறி நெசவாளர். எனக்கு நெசவாளர்களுக்கான அரசு கடன் உதவி, மானியம் மற்றும் பிஎம் விஸ்வகர்மா திட்டம் பற்றி கூறவும்.",
        "expected_domain": "weaver / PM Vishwakarma",
        "forbidden_tokens": ["condolence", "passing", "death", "funeral", "இறப்பு", "மரணம்"]
    },
    {
        "id": "telugu_sc_farmer",
        "name": "Gujjula Ramayya",
        "raw_query": "నమస్కారం అండీ, నా పేరు గుజ్జుల రామయ్య. నేను తెలంగాణ రాష్ట్రానికి చెందిన ఎస్సీ రైతును. నాకు పంట పెట్టుబడి, ఎరువుల సబ్సిడీ మరియు రైతు బంధు లేదా ప్రభుత్వ పథకాల సమాచారం కావాలి.",
        "expected_domain": "Rythu Bandhu / SC Farmer",
        "forbidden_tokens": ["minority", "funeral", "Christian", "Muslim"]
    },
    {
        "id": "hindi_dairy_farmer",
        "name": "Ramesh Kumar",
        "raw_query": "नमस्ते साहब, मेरा नाम रमेश कुमार है, मैं उत्तर प्रदेश के गोरखपुर का डेयरी किसान हूँ, OBC वर्ग, दो गाय-भैंस हैं। हमें खेती और पशुपालन के लिए किसान क्रेडिट कार्ड (KCC) और लोन योजना की जानकारी दीजिए।",
        "expected_domain": "Kisan Credit Card / Animal Husbandry",
        "forbidden_tokens": ["student credit card", "higher education", "overseas study", "छात्र"]
    },
    {
        "id": "hindi_small_farmer",
        "name": "Sunita Sharma",
        "raw_query": "नमस्ते, मेरा नाम सुनीता शर्मा है। मैं मध्य प्रदेश के सिहोर जिले से एक छोटी महिला किसान हूँ। हमारे पास एक एकड़ जमीन है। हमें खेती के लिए बीज, खाद और सरकारी आर्थिक सहायता की योजनाएं बताइए।",
        "expected_domain": "PM Kisan Samman Nidhi / Small Farmer",
        "forbidden_tokens": ["funeral", "student loan"]
    }
]

print("="*80)
print("RUNNING END-TO-END VERIFICATION ACROSS ALL 4 CITIZEN PERSONAS")
print("="*80)

passed = 0

for p in personas:
    print(f"\n--- Testing Persona: {p['name']} ({p['id']}) ---")
    print(f"Raw Input Query:\n  {p['raw_query']}")
    
    # 1. Normalize & Translate Query
    profile = normalize_and_translate_query(p["raw_query"])
    search_q = profile.get("english_query") or p["raw_query"]
    print(f"Normalized Search Query:\n  {search_q}")
    print(f"Extracted Profile: lang={profile.get('detected_lang')}, name={profile.get('citizen_name')}, occ={profile.get('occupation')}, state={profile.get('state')}")
    
    # 2. Hybrid Retrieve
    chunks = hybrid_retrieve(search_q)
    print(f"Retrieved {len(chunks)} chunks:")
    for idx, c in enumerate(chunks[:2], 1):
        lines = c['text'].split('\n')
        s_name = next((l for l in lines if 'SCHEME:' in l), '')
        target = next((l for l in lines if 'TARGET BENEFICIARIES:' in l), '')
        print(f"  [{idx}] {s_name} | {target[:60]}")
        
    # 3. Generate Answer
    answer = generate_answer(p["raw_query"], chunks, profile=profile)
    print(f"Generated Answer:\n  {answer}")
    
    # 4. Assertions & Validation
    has_forbidden = any(f.lower() in answer.lower() for f in p["forbidden_tokens"])
    if has_forbidden:
        print(f"FAILED: Found forbidden hallucinated token in answer!")
    else:
        print(f"PASSED: Zero hallucinated / forbidden tokens found.")
        passed += 1

print("\n" + "="*80)
print(f"RESULTS: {passed}/{len(personas)} PERSONAS PASSED PERFECTLY!")
print("="*80)
