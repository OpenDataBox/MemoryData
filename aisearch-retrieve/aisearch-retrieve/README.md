# Linkpoket AI 검색 — retrieve(자료 조회·선별) 코드 발췌

원본: Linkrew-backEnd `src/main/java/com/linkpoket/source/domain/aisearch/` (develop 48b35093 기준, 2026-08-19 추출)

## 포함 범위 — 질문이 들어와서 "참고 자료"가 확정되기까지
- service/  : 파이프라인 오케스트레이션(AiSearchService·Router), 프롬프트 조립(MdComposer),
              LLM 호출(AiSearchLlmCaller), 폴더/페이지 설명 생성(Description*, *Builder)
- event/    : 링크 본문 추출(LinkBodyExtraction*), 폴더/페이지 메타 갱신 리스너
- entity/   : ai_item_body(추출 본문), ai_folder_meta / ai_page_meta(검색용 요약)
- repository/, dto/, error/, util/(프롬프트 인젝션 방어)

## 제외
- 대화 저장(AiConversation*), 사용량 로그(AiSearchUsageLog*), 쿼터(QuotaChecker), 컨트롤러

## 참고 (외부 의존)
- 폴더/아이템 실조회는 folder·link·note·file 도메인의 DataAccess를 호출함 (이 zip엔 미포함)
- 알려진 이슈·개선안은 AI_SEARCH_UPGRADE_SPEC.md 참조
