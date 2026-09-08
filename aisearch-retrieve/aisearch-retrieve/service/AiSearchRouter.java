package com.linkpoket.source.domain.aisearch.service;

import java.util.List;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.MessageSource;
import org.springframework.context.i18n.LocaleContextHolder;
import org.springframework.stereotype.Component;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 실시간 AI 지식 검색의 2단계(폴더 탐색), 3단계(아이템 추출), 4단계(답변 생성)를 순차 결합하는 라우팅 컴포넌트 클래스입니다.
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class AiSearchRouter {

	private final MdComposer mdComposer;
	private final AiSearchLlmCaller llmCaller;
	private final MessageSource messageSource;

	/**
	 * 복수 페이지 하위 후보 아이템들을 조립 및 정렬하고 단계별 LLM 호출을 이어붙여 최종 답변 DTO를 리턴합니다.
	 * 답변 생성 단계에서 일반 예외가 발생할 경우 부분 성공(fallback)으로 간주하여 다국어 설명 메시지와 참고 출처 리스트를 반환합니다.
	 *
	 * @param query             사용자 질문문
	 * @param accessiblePageIds 권한 검증이 끝난 접근 가능 페이지 ID 목록
	 * @param memberId          회원 식별자
	 * @param history           같은 대화 세션의 이전 질문/답변 이력 (단발 검색이면 빈 목록)
	 * @return 검색 답변 DTO
	 */
	public AiSearchResult route(
		final String query,
		final List<Long> accessiblePageIds,
		final Long memberId,
		final List<ConversationTurn> history
	) {
		final long start = System.currentTimeMillis();
		final AiSearchContext context = new AiSearchContext("", memberId);

		// 2단계: 폴더 탐색
		final PromptBundle folderBundle = mdComposer.composeFolderListPrompt(accessiblePageIds);
		final List<Long> candidateFolders = llmCaller.selectFolders(
			query, folderBundle.folderListPrompt(), context, history);

		// 3단계: 관련 아이템 추출 (1차 시도 — LLM이 좁힌 폴더 범위 내에서 탐색)
		ItemBodyBundle itemBundle = searchItems(query, candidateFolders, context, history);

		// 에이전틱 루프(재탐색, 최대 1회 추가 시도 — 총 2회):
		// 1차 시도에서 근거 자료를 하나도 못 찾으면, 폴더 선택 단계를 건너뛰고 이 페이지의
		// 전체 후보 폴더로 범위를 넓혀 아이템 탐색만 다시 수행한다. LLM이 처음에 관련 없다고
		// 판단해 걸러낸 폴더 중 실제로는 관련 있는 것이 있을 수 있기 때문이다.
		if (itemBundle.sources().isEmpty() && !folderBundle.folderIds().isEmpty()
			&& !candidateFolders.equals(folderBundle.folderIds())) {
			log.info("[AiSearch] agentic retry triggered — broadening folder scope. narrowed={}, full={}, memberId={}",
				candidateFolders.size(), folderBundle.folderIds().size(), memberId);
			itemBundle = searchItems(query, folderBundle.folderIds(), context, history);
		}

		final boolean noMaterialsFound = itemBundle.sources().isEmpty();
		if (noMaterialsFound) {
			log.info("[AiSearch] no item selected after retry. memberId={}", memberId);
		}

		// 4단계: 답변 생성
		// 저장 자료가 없어도 4단계를 생략하지 않는다 — 개념 설명 등 LLM 본연의 기능은 계속 제공하되,
		// "이번 질문엔 참고할 저장 자료가 없다"는 사실을 프롬프트로 알려 사실 관계를 지어내지 않게 한다.
		String answer;
		boolean isSuccess = true;
		try {
			answer = llmCaller.generateAnswer(query, itemBundle.itemsPrompt(), context, history, noMaterialsFound);
		} catch (final Exception e) {
			log.warn("[AiSearch] routing 4-stage generation failed. fallback to partial success. cause={}", e.toString());
			isSuccess = false;
			final Object[] args = new Object[]{itemBundle.sources().size()};
			answer = messageSource.getMessage("ai.search.partial_success", args, LocaleContextHolder.getLocale());
		}

		final long latency = System.currentTimeMillis() - start;
		return new AiSearchResult(answer, itemBundle.sources(), context.getTotalTokens(), latency, isSuccess);
	}

	/**
	 * 주어진 폴더 범위 내에서 3단계(아이템 선택)를 수행합니다.
	 * 1차 시도와 에이전틱 재탐색이 동일한 로직을 공유하기 위해 분리했습니다.
	 *
	 * @param query    사용자 질문문
	 * @param folderIds 탐색 대상 폴더 식별자 목록
	 * @param context  요청 실행 컨텍스트
	 * @param history  이전 대화 이력
	 * @return 선택된 아이템만 남은 번들 (없으면 빈 번들)
	 */
	private ItemBodyBundle searchItems(
		final String query,
		final List<Long> folderIds,
		final AiSearchContext context,
		final List<ConversationTurn> history
	) {
		final ItemBodyBundle candidateBundle = mdComposer.composeItemsPrompt(folderIds);
		final List<Long> selectedItemIds = llmCaller.selectItems(query, candidateBundle.itemsPrompt(), context, history);
		return candidateBundle.filterBy(selectedItemIds);
	}
}