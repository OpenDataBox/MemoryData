package com.linkpoket.source.domain.aisearch.service;

import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

import lombok.Getter;

/**
 * 단일 AI 검색 요청의 실행 상태를 담는 컨텍스트입니다.
 */
@Getter
public class AiSearchContext {

	private final String traceId;
	private final Long memberId;
	private final Instant startedAt = Instant.now();
	private final Map<Integer, Long> stageLatency = new HashMap<>();

	private int totalCalls;
	private long totalTokens;

	public AiSearchContext(final String traceId, final Long memberId) {
		this.traceId = traceId;
		this.memberId = memberId;
	}

	public void record(final int stage, final long tokens, final long elapsedMs) {
		this.totalCalls++;
		this.totalTokens += tokens;
		this.stageLatency.merge(stage, elapsedMs, Long::sum);
	}

	public long getStageLatency(final int stage) {
		return stageLatency.getOrDefault(stage, 0L);
	}

	public Duration elapsed() {
		return Duration.between(startedAt, Instant.now());
	}
}
