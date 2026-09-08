package com.linkpoket.source.domain.aisearch.service;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.linkpoket.source.domain.aisearch.dto.response.AiSearchSourceDto;

/**
 * 3단계 LLM 질문 아이템 추천용 프롬프트와 선정 완료된 검색 결과 출처 DTO 목록을 묶어 관리하는 record 객체입니다.
 *
 * @param itemsPrompt 마크다운 형태의 아이템 리스트 본문 프롬프트
 * @param sources     검색 결과로 노출할 출처 목록 DTO 리스트
 * @param itemLines   아이템 식별자별 마크다운 본문 라인 (선택 결과에 맞춰 프롬프트를 다시 조립하기 위한 원본)
 */
public record ItemBodyBundle(
	String itemsPrompt,
	List<AiSearchSourceDto> sources,
	Map<Long, String> itemLines
) {

	/**
	 * 3단계 LLM이 실제로 선택한 아이템만 남긴 새로운 번들을 만듭니다.
	 * 답변 생성 프롬프트와 사용자에게 노출할 출처를 같은 기준으로 좁혀 둘 사이의 불일치를 없앱니다.
	 *
	 * @param selectedItemIds LLM이 선택한 아이템 식별자 목록
	 * @return 선택된 아이템만 담긴 번들 (선택 결과가 비어 있으면 빈 번들)
	 */
	public ItemBodyBundle filterBy(final List<Long> selectedItemIds) {
		if (selectedItemIds == null || selectedItemIds.isEmpty()) {
			return new ItemBodyBundle(emptyPrompt(), List.of(), Map.of());
		}
		final Set<Long> selected = Set.copyOf(selectedItemIds);

		final Map<Long, String> filteredLines = new LinkedHashMap<>();
		itemLines.forEach((itemId, line) -> {
			if (selected.contains(itemId)) {
				filteredLines.put(itemId, line);
			}
		});
		if (filteredLines.isEmpty()) {
			return new ItemBodyBundle(emptyPrompt(), List.of(), Map.of());
		}

		final List<AiSearchSourceDto> filteredSources = sources.stream()
			.filter(source -> selected.contains(source.itemId()))
			.toList();

		final StringBuilder builder = new StringBuilder("### 관련 아이템 목록\n");
		filteredLines.values().forEach(builder::append);

		return new ItemBodyBundle(
			MdComposer.USER_DATA_START + builder + MdComposer.USER_DATA_END,
			filteredSources,
			filteredLines
		);
	}

	/**
	 * 선택된 아이템이 하나도 없을 때 사용할 빈 프롬프트를 만듭니다.
	 *
	 * @return 관련 아이템이 없음을 알리는 프롬프트 문자열
	 */
	private static String emptyPrompt() {
		return MdComposer.USER_DATA_START + "### 관련 아이템 목록\n(관련 아이템이 없습니다)\n" + MdComposer.USER_DATA_END;
	}
}
