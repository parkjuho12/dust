# n8n 워크플로우

## 개요

n8n을 활용하여 사용자 입력 → 온톨로지 슬롯 수집 → ML 예측 → REG-LEG 추론 → 행동 조언 출력의 전체 파이프라인을 구현했습니다. 비교 실험을 위해 REG-LEG 워크플로우와 LLM 자유 생성 워크플로우 두 가지를 구축했습니다.

---

## 1. REG-LEG 워크플로우 (`PM10_REG-LEG_Advisory1_gemma_v2.json`)

### 전체 흐름

```
사용자 입력
→ 세션 상태 조회 (DataTable)
→ Fact Collector (온톨로지 슬롯 수집)
→ State Completeness Assessment (슬롯 완성도 체크)
→ 미완성: 추가 질문 → 사용자 재입력
→ 완성: FastAPI /predict 호출 (ML 예측)
→ Regulatory Rule Retrieval (REG 규칙 조회)
→ Deterministic Logical Evaluation (LEG 추론)
→ Judgment Explanation Generation (leg_message 생성)
→ Judgment Outcome Logging (구글 시트 로깅)
→ Judgment Result Delivery (사용자 답변)
→ Session Reset (세션 초기화)
```

### 노드 설명

| 노드 | 역할 |
|------|------|
| User Input Reception | 사용자 채팅 입력 수신 |
| Persistent Session State Retrieval | DataTable에서 세션 상태 조회 |
| Session State Preprocessing | oldFacts, oldAsked 파싱 |
| Ontology-Guided State Representation Controller | 사용자 입력에서 슬롯 추출 (Fact Collector) |
| Structured Output Parser | LLM 출력을 JSON 구조로 파싱 |
| State Evidence Accumulation | 기존 facts와 신규 facts 병합 |
| State Completeness Assessment & Query Generation | 슬롯 완성도 체크 및 추가 질문 생성 |
| Upsert row(s) | DataTable에 세션 상태 저장 |
| Ready for Prediction | 슬롯 완성 여부 분기 |
| HTTP Request | FastAPI `/predict` 호출 |
| Prediction Context Restore | 예측 결과와 슬롯 컨텍스트 병합 |
| Regulatory Rule Retrieval | PM10 등급 + 사용자 컨텍스트 기반 규칙 조회 |
| Deterministic Logical Evaluation (LEG) | 규칙 기반 행동 결정 |
| Judgment Explanation Generation | 최종 답변 메시지 생성 (leg_message) |
| Judgment Outcome Logging | 구글 시트에 결과 로깅 |
| Judgment Result Delivery | 사용자에게 답변 전달 |
| Session Reset | 세션 초기화 후 DataTable upsert |

### 온톨로지 슬롯

| 슬롯 | 허용 값 |
|------|---------|
| user_role | PARENT, STUDENT, TEACHER, WORKER, UNKNOWN |
| activity_type | COMMUTE, PE_CLASS, OUTDOOR_CLASS, OUTDOOR_WORK, OUTDOOR_PLAY, EXERCISE, UNKNOWN |
| sensitive_group | TRUE, FALSE, UNKNOWN |
| region | 서울 OO구 형태 문자열, UNKNOWN |

### REG 규칙 (rule_id)

| rule_id | 조건 | action |
|---------|------|--------|
| REG_PM_01 | VERY_BAD + 민감군 | INDOOR_RECOMMENDED, CONSULT_DOCTOR_IF_OUTDOOR, WEAR_MASK |
| REG_PM_02 | VERY_BAD + 일반 | LIMIT_OUTDOOR_ACTIVITY, WEAR_MASK |
| REG_PM_03 | VERY_BAD + 야외활동/운동 | STOP_OUTDOOR_ACTIVITY, WEAR_MASK |
| REG_PM_04 | VERY_BAD + WORKER + OUTDOOR_WORK | PROTECTIVE_MEASURES_REQUIRED, WEAR_MASK |
| REG_PM_05 | BAD + 민감군 | LIMIT_OUTDOOR_ACTIVITY, WEAR_MASK |
| REG_PM_06 | BAD + 일반 | LIMIT_OUTDOOR_ACTIVITY, WEAR_MASK |
| REG_PM_07 | BAD + TEACHER + 야외수업 | SWITCH_TO_INDOOR |
| REG_PM_08 | BAD + PARENT + OUTDOOR_PLAY | LIMIT_CHILD_OUTDOOR |
| REG_PM_09 | NORMAL + 민감군 | CAUTION_SENSITIVE |
| REG_PM_09A | NORMAL + 일반 | CAUTION |
| REG_PM_10 | GOOD | NORMAL_ACTIVITY |

### 구글 시트 컬럼

| 컬럼 | 설명 |
|------|------|
| test_id | sessionId |
| timestamp | 실행 시각 |
| input_text | 사용자 입력 문장 |
| user_role | 추출된 역할 |
| activity_type | 추출된 활동 유형 |
| sensitive_group | 민감군 여부 |
| region | 추출된 지역 |
| prediction | PM10 예측 등급 |
| pm_level_code | PM10 등급 코드 |
| rule_key | 적용된 rule_id |
| action | 결정된 행동 코드 |
| reg_status | 규칙 적용 상태 |
| reason_codes | 행동 근거 코드 |
| explanation | leg_message (최종 답변) |

---

## 2. LLM 자유 생성 워크플로우 (`PM10_LLM_Baseline.json`)

### 전체 흐름

```
사용자 입력
→ FastAPI /predict 호출 (ML 예측)
→ AI Agent (GPT-4.1-mini, PM10 등급 + 사용자 입력 기반 자유 답변)
→ 구글 시트 로깅
→ 사용자 답변
```

### 구글 시트 컬럼

| 컬럼 | 설명 |
|------|------|
| input_text | 사용자 입력 문장 |
| Ai action | LLM 자유 생성 답변 |

---

## 3. 비교 실험 설계

### 목적
REG-LEG 온톨로지 기반 추론과 LLM 자유 생성 방식의 답변 품질 비교

### 입력
동일한 30개 테스트 케이스를 두 워크플로우에 각각 입력 (`test_cases.md` 참고)

### 평가 기준

| 기준 | 설명 | 점수 |
|------|------|------|
| 정확성 | 환경부 미세먼지 행동요령과 일치하는가 | 1~5 |
| 설명가능성 | 근거(rule_id, action)가 명시되는가 | 1~5 |
| 일관성 | 같은 입력에 항상 같은 답변이 나오는가 | 1~5 |
| 맥락 반영 | 사용자 역할/활동/민감도가 반영되는가 | 1~5 |
