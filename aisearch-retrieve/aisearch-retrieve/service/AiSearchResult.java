package com.linkpoket.source.domain.aisearch.service;

import java.util.List;
import com.linkpoket.source.domain.aisearch.dto.response.AiSearchSourceDto;

/**
 * AI 검색 라우팅 내부 결과를 나타내는 record 객체입니다.
 *
 * @param answer    생성된 답변
 * @param sources   참조된 출처 리스트
 * @param tokens    소모된 총 토큰 수
 * @param latencyMs 소요된 밀리초
 * @param isSuccess 성공 여부
 */
public record AiSearchResult(
	String answer,
	List<AiSearchSourceDto> sources,
	long tokens,
	long latencyMs,
	boolean isSuccess
) {
}
