"""생성 모델 매핑/폴백 로직 검증 (API 키 불필요, 로직만 테스트)
app.py의 MODEL_MAP / resolve_model / generate_answer 폴백 흐름을 재현하여 확인한다.
"""

MODEL_MAP = {
    "gemini-1.5-pro (고정밀)": "gemini-1.5-pro",
    "gemini-1.5-flash (고속)": "gemini-1.5-flash",
}
GENERATION_FALLBACKS = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-flash-latest", "gemini-2.0-flash"]


def resolve_model(selection, available):
    primary = MODEL_MAP[selection]
    if not available or f"models/{primary}" in available:
        return primary
    for c in GENERATION_FALLBACKS:
        if f"models/{c}" in available:
            return c
    return primary


# 1) UI 선택값 → 정확한 정식 모델명 1:1 매핑 확인
assert resolve_model("gemini-1.5-pro (고정밀)", set()) == "gemini-1.5-pro"
assert resolve_model("gemini-1.5-flash (고속)", set()) == "gemini-1.5-flash"
print("✅ UI 선택값 → 정식 모델명(gemini-1.5-pro / gemini-1.5-flash) 1:1 매핑 정상")

# 2) 계정에서 1.5 계열 제공 시 그대로 사용
avail = {"models/gemini-1.5-pro", "models/gemini-1.5-flash", "models/gemini-2.0-flash"}
assert resolve_model("gemini-1.5-pro (고정밀)", avail) == "gemini-1.5-pro"
print("✅ 사용 가능 목록에 1.5-pro 존재 시 그대로 호출")

# 3) 1.5-pro 미제공 계정 → 사용 가능한 폴백(1.5-flash) 선택, 2.5-pro는 절대 선택 안 됨
avail2 = {"models/gemini-1.5-flash", "models/gemini-2.0-flash", "models/gemini-2.5-pro"}
r = resolve_model("gemini-1.5-pro (고정밀)", avail2)
assert r == "gemini-1.5-flash" and r != "gemini-2.5-pro"
print(f"✅ 1.5-pro 미제공 시 폴백 선택: {r} (gemini-2.5-pro 미사용 확인)")

# 4) generate_answer 재시도 흐름: 첫 모델 404 → 다음 후보 성공
calls = []
def fake_call(name):
    calls.append(name)
    if name == "gemini-1.5-pro":
        raise Exception("404 This model is no longer available to new users.")
    return f"answer from {name}"

def generate_with_fallback(model_name):
    for name in [model_name] + [m for m in GENERATION_FALLBACKS if m != model_name]:
        try:
            return fake_call(name)
        except Exception as e:
            msg = str(e).lower()
            if not any(k in msg for k in ["404", "not found", "no longer available", "deprecated", "not supported"]):
                raise
    raise RuntimeError("no model")

ans = generate_with_fallback("gemini-1.5-pro")
assert ans == "answer from gemini-1.5-flash" and calls == ["gemini-1.5-pro", "gemini-1.5-flash"]
print(f"✅ 404 발생 시 재시도 순서 정상: {calls} → '{ans}'")

# 5) app.py에 gemini-2.5-pro가 기본 호출 경로에 남아있지 않은지 정적 검사
src = open("/home/ubuntu/research-assistant/app.py", encoding="utf-8").read()
assert '"gemini-2.5-pro"' not in src and "'gemini-2.5-pro'" not in src
print("✅ app.py 내 gemini-2.5-pro 하드코딩 제거 확인")

print("\n🎉 생성 모델 매핑/폴백 로직 테스트 전부 통과")
