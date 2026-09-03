package com.linkpoket.source.domain.aisearch.dto.response;

import java.util.List;

/**
 * AI 검색 응답 DTO입니다.
 */
public record AiSearchResponse(
	String answer,
	List<AiSearchSourceDto> sources,
	AiSearchUsageDto usage
) {
}
