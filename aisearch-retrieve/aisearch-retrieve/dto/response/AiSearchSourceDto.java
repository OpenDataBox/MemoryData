package com.linkpoket.source.domain.aisearch.dto.response;

import com.linkpoket.source.global.json.SnowflakeId;

import lombok.Builder;

/**
 * AI 검색 답변의 근거로 사용된 아이템 정보입니다.
 */
@Builder
public record AiSearchSourceDto(
	@SnowflakeId Long itemId,
	String itemType,
	String title,
	String url
) {
}
