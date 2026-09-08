# 기준선 개발 문항 진단

LoCoMo 고정 개발 70문항: 공식 category F1 **0.4755517**, 일반 token F1 0.4394964.
평균 읽기 2038.6 tokens, 대화당 저장 평균9372.7 tokens. 출처 ID recall 평균0.6733.
`unknown` 응답12개, F1<0.5 문항35개, 이 중 출처 ID가 gold 근거 전체를 포함하지 않은 문항18개.
출처 ID 포함은 사실 보존이나 entailment의 증명이 아니다.

관찰한 실패 양상:

- 세부 사건 혼동: clay pot 질문에서 gold의 dog-face cup 대신 flower-design bowl을 답함.
- 지나치게 긴 답: 팀을 발전시키는 동기를 묻자 tough opponents 외에 다수의 다른 동기를 나열해 F1 저하.
- 상대/절대 날짜 표현 차이: November2023에 next month, A year ago에 절대 날짜를 답한 예시.
- open-domain 문항은0.0095 F1(5문항)로 가장 낮음. 음악 장르·장소·국가에 대한 추론/검색 문제가 섞여 있음.

다음 구성은 참여자·행동·대상·이유·시간을 같은 사건 단위에 묶고, 원문에 있으나 메모리에서 빠진
상세 사실을 재검사한다. 읽기 모델·프롬프트·retrieval·2048-token cap은 유지하므로 구성 변경의
효과를 분리한다. 검증30문항의 gold/실패 예시는 이 진단에 사용하지 않았다.
