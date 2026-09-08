# 고정한 parent_single 정책의 새 100문항 checkpoint

이 계획은 identity 통합 실험의 새로운 개발 F1을 보기 전에 고정한다. 선택 방법은 이미 확정된 `s_parent_single_2000` (개발 F1 59.27437)이다. 비교군은 초기 seed와 r40이다. 진행 중인 identity 대조에서 새로운 최고가 나오더라도 이번 audit의 정책이나 비교군은 바꾸지 않는다.

별도 고정 manifest의 100개 질문 ID는 최초 평가 100개와 겹치지 않는다. 대상은 conv-42/47/50이며 최초 pilot에서 일부 질문을 이미 사용한 대화다. **새 질문의 checkpoint이며 미사용 대화 또는 외부 데이터셋 검증이 아니다.** Audit 결과를 후속 refinement 선택·규칙 수정에 사용하지 않는다.

준비 단계에서는 source 세션만 함수에 넘긴다. 기존 seed에 r06 날짜 고정, r12 일반 대화 필터, r40 네 turn 병합을 동일하게 적용한다. 기존 질문 생성 프롬프트와 10-turn/2-overlap 창, literal-span compiler, 세션당 2개·seed 20260908의 fit/session-audit 분리를 그대로 쓴다. 출처 형식을 생성기가 누락해도 전체 생성 입력창과 literal answer 위치로 연결하는 기존 compiler를 사용한다. 별도 source session audit는 구성에 쓰지 않는다.

다음 독립 GPU 프로세스에서 동일 NLL 정책·토큰 연결·prefix-cache off·seed 20260908로 source-fit 전체의 단일/쌍/전체를 계산하고 실제 control 답을 생성한다. 마지막 GPU 프로세스는 원래 benchmark runtime과 seed 20260907로 돌아가 기존 parent_single +2000 구성만 적용한다. 이 정책은 검증 옵션을 선택에 쓰지 않으므로 불필요한 verifier 생성은 생략한다. 기존 개발 7개 대화의 r40과 최종 parent 메모리가 **내용·구조까지 완전히 동일하게 재구성되는 gate**로 이 생략의 동등성을 확인한다. 개발 기준 F1도 정확히 재현해야 한다.

모든 방법·세 대화의 메모리 hash와 source 구성 입력을 고정한 뒤, 새 질문 QA를 읽어 3개 방법을 각각 100문항 평가한다. 같은 reader, BM25+dense RRF, 실제 2048-token 읽기 상한을 사용한다. 저장은 검색 키와 원문 payload를 모두 계산한다. 평가 후 방법을 고르는 절차는 없다. 대화별 paired bootstrap 구간은 대화 3개뿐인 기술적 참고치로만 보고한다.

단일 RTX 4080S를 공유하므로 identity 실험이 정상 종료하고 GPU 계산 프로세스가 없음을 확인한 뒤 준비→확률 측정→평가 프로세스를 순서대로 실행한다. 이 checkpoint 종료와 지속 refinement 목표 종료는 별개다.

데이터 접근 문구의 정확한 범위: 준비 코드는 원본 JSON 전체를 파싱하고 대상 대화의 source 세션만 구성 함수로 넘긴다. QA 바이트가 프로세스 메모리에 전혀 로드되지 않았다는 주장은 아니다. QA 미사용은 대상 벤치마크 QA 필드를 구성·선택 입력으로 사용하지 않았다는 의미다. 새 문항을 실제 평가 입력으로 추출하는 단계는 메모리 고정 뒤다.
