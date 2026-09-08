package com.linkpoket.source.domain.aisearch.dto.request;

import com.linkpoket.source.global.json.SnowflakeId;

import jakarta.validation.constraints.NotBlank;

/**
 * AI 검색 요청 DTO입니다.
 */
public record AiSearchRequest(
	@NotBlank(message = "검색어는 필수입니다.") String query,
	@SnowflakeId Long pageId
) {
}
